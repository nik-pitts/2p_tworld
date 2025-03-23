#!/usr/bin/env python3
"""
Simplified training script for RL Agent in Chip's Challenge environment
with straightforward wandb integration
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor


# Import the custom environment and BC model
from src.gym_env import GymEnv
from src.models import BehaviorCloningModel


class BCFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that handles additional RL observations
    while using the BC model for processing the original inputs
    """

    def __init__(self, observation_space, bc_model, features_dim=128):
        super(BCFeaturesExtractor, self).__init__(observation_space, features_dim)

        # Store BC model (without the output layer)
        self.bc_layers = nn.Sequential(*list(bc_model.fc)[:-1])

        # Get BC model input size and total observation size
        self.bc_input_size = bc_model.fc[0].in_features
        self.observation_size = observation_space.shape[0]
        self.extra_hidden_neurons = 16

        # Check if we have additional observations
        if self.observation_size > self.bc_input_size:
            # Calculate extra input size
            self.extra_size = self.observation_size - self.bc_input_size

            # Create a processor for the extra inputs
            self.extra_processor = nn.Sequential(
                nn.Linear(self.extra_size, self.extra_hidden_neurons),
                nn.ReLU(),
                nn.Linear(self.extra_hidden_neurons, self.extra_hidden_neurons),
                nn.ReLU(),
            )

            # Create a combiner network that merges BC features and extra features
            self.combiner = nn.Linear(
                128 + self.extra_hidden_neurons, features_dim
            )  # BC output is 128, extra is self.extra_hidden_neurons
        else:
            # No extra features, just pass BC features through
            self.extra_processor = None
            self.combiner = nn.Identity()

    def forward(self, observations):
        """Process observations with BC model and handle extra features"""
        # Process original BC inputs
        bc_inputs = observations[:, : self.bc_input_size]
        bc_features = self.bc_layers(bc_inputs)

        # If we have extra inputs, process them and combine
        if self.extra_processor is not None:
            extra_inputs = observations[:, self.bc_input_size :]
            extra_features = self.extra_processor(extra_inputs)

            # Combine features
            combined = torch.cat([bc_features, extra_features], dim=1)
            return self.combiner(combined)
        else:
            # No extra features, just return BC features
            return bc_features


class WandbCallback(BaseCallback):
    """Callback for wandb logging with minimal overhead"""

    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_count = 0

    def _on_step(self):
        # Log any available metrics
        if len(self.model.ep_info_buffer) > 0 and len(self.ep_rewards) < len(
            self.model.ep_info_buffer
        ):
            # Get new episode infos
            new_infos = list(self.model.ep_info_buffer)[len(self.ep_rewards) :]

            for info in new_infos:
                self.ep_rewards.append(info["r"])
                self.ep_lengths.append(info["l"])
                self.ep_count += 1

                # Log individual episode stats
                wandb.log(
                    {
                        "episode": self.ep_count,
                        "episode_reward": info["r"],
                        "episode_length": info["l"],
                    }
                )

            # Log rolling averages
            wandb.log(
                {
                    "avg_reward": np.mean(self.ep_rewards[-100:]),
                    "avg_length": np.mean(self.ep_lengths[-100:]),
                    "episodes": self.ep_count,
                    "timesteps": self.num_timesteps,
                }
            )

        return True


def save_bc_compatible_model(ppo_model, bc_model, output_path):
    """
    Save the trained policy in a format compatible with the original BC model
    Only keeping the weights for the original BC inputs
    """
    # Create a new BC model with the same architecture
    compatible_model = BehaviorCloningModel(bc_model.fc[0].in_features, 4)

    # Check if we're using a custom feature extractor
    if hasattr(ppo_model.policy, "features_extractor") and isinstance(
        ppo_model.policy.features_extractor, BCFeaturesExtractor
    ):
        # Copy BC layers from feature extractor
        with torch.no_grad():
            for i in range(len(compatible_model.fc) - 1):  # Skip output layer
                if isinstance(compatible_model.fc[i], nn.Linear):
                    idx = i
                    bc_layer = list(ppo_model.policy.features_extractor.bc_layers)[idx]
                    compatible_model.fc[i].weight.copy_(bc_layer.weight)
                    compatible_model.fc[i].bias.copy_(bc_layer.bias)

            # Copy output layer (action head) from policy network
            compatible_model.fc[-1].weight.copy_(ppo_model.policy.action_net.weight)
            compatible_model.fc[-1].bias.copy_(ppo_model.policy.action_net.bias)
    else:
        # Direct copy from policy if not using custom extractor
        print("Provide feature extractor")

    # Save the compatible model
    torch.save(compatible_model.state_dict(), output_path)
    print(f"Saved BC-compatible model to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train RL agent for Chip's Challenge")

    # Environment settings
    parser.add_argument("--level", type=int, default=0, help="Level index to train on")
    parser.add_argument(
        "--p1-model",
        type=str,
        default="./model/lv1_bc_model_3.8.pth",
        help="Path to BC model for player 1 (will be trained with RL)",
    )
    parser.add_argument(
        "--p2-model",
        type=str,
        default="./model/lv1_bc_model_3.8.pth",
        help="Path to BC model for player 2 (fixed)",
    )

    # Training settings
    parser.add_argument(
        "--timesteps", type=int, default=200000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Wandb settings
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (defaults to timestamp)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="chips-challenge-rl",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./model", help="Output directory"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Set experiment name
    if args.exp_name is None:
        args.exp_name = f"compatible_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    set_random_seed(args.seed)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        config={
            "algorithm": "PPO",
            "level": args.level,
            "p1_model": args.p1_model,
            "p2_model": args.p2_model,
            "timesteps": args.timesteps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    )

    # Load the BC model to extract architecture and weights
    bc_input_size = 191  # BC model's input size
    bc_model = BehaviorCloningModel(bc_input_size, 4)
    bc_model.load_state_dict(torch.load(args.p1_model, weights_only=True))
    bc_model.eval()

    # Create environment
    env = GymEnv(
        level_index=args.level,
        p1_bc_model_path=args.p1_model,
        p2_bc_model_path=args.p2_model,
    )

    env = Monitor(env)

    # Get observation size from environment
    obs_size = env.observation_space.shape[0]
    print(
        f"Environment observation size: {obs_size}, BC model input size: {bc_input_size}"
    )

    # Configure custom feature extraction if needed
    if obs_size > bc_input_size:
        print(
            f"Using custom feature extractor to handle additional {obs_size - bc_input_size} features"
        )
        policy_kwargs = {
            "features_extractor_class": BCFeaturesExtractor,
            "features_extractor_kwargs": {"bc_model": bc_model},
            "net_arch": dict(pi=[128, 128, 128], vf=[128, 128, 128]),
        }
    else:
        # If observation size matches BC input, use standard architecture
        print("Using standard policy network (observations match BC input size)")
        policy_kwargs = {
            "net_arch": dict(pi=[128, 128, 128], vf=[128, 128, 128]),
            "activation_fn": nn.ReLU,
        }

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
        batch_size=64,
        ent_coef=0.01,  # Encourage exploration
        clip_range=0.2,
    )

    # Create wandb callback
    wandb_callback = WandbCallback()

    try:
        # Train model
        print(f"Starting training for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps, callback=wandb_callback)

        # Save models
        # 1. Save in SB3 format
        # sb3_model_path = os.path.join(args.output_dir, f"{args.exp_name}_sb3.zip")
        # model.save(sb3_model_path)

        # 2. Save in BC-compatible format
        bc_compatible_path = os.path.join(
            args.output_dir, f"{args.exp_name}_bc_compatible.pth"
        )
        save_bc_compatible_model(model, bc_model, bc_compatible_path)

        # Final evaluation
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )

        print(f"Final evaluation: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        wandb.log({"final_mean_reward": mean_reward, "final_std_reward": std_reward})

        # Log models to wandb
        # wandb.save(sb3_model_path)
        # wandb.save(bc_compatible_path)

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up
        env.close()
        wandb.finish()


main()
