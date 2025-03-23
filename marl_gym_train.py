#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning (MARL) training script for Chip's Challenge
Trains two agents simultaneously to cooperate using pretrained BC models as starting points
Simplified version using model.learn() from Stable Baselines3
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import wandb
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from src.models import BehaviorCloningModel


#######################################
# Agent Wrapper Classes
#######################################


class AlternatingAgentWrapper(gym.Wrapper):
    """
    Wrapper for training agents in alternating fashion.
    When training agent 1, agent 2's actions come from a fixed policy.
    When training agent 2, agent 1's actions come from a fixed policy.
    """

    def __init__(self, env, agent_id, fixed_policy=None):
        super().__init__(env)
        self.agent_id = agent_id
        self.agent_key = f"agent{agent_id}"
        self.other_agent_key = f"agent{3 - agent_id}"  # 3-1=2, 3-2=1
        self.fixed_policy = fixed_policy  # Policy for the other agent

        # Store the latest observation for both agents
        self.latest_obs = None

        # Define observation and action spaces for this agent
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space[self.agent_key]
            self.action_space = env.action_space[self.agent_key]
        else:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

    def set_fixed_policy(self, policy):
        """Set the policy for the other agent"""
        self.fixed_policy = policy

    def reset(self, **kwargs):
        """Reset the environment and return only this agent's observation"""
        self.latest_obs, info = self.env.reset(**kwargs)
        return self.latest_obs[self.agent_key], info

    def step(self, action):
        """Step the environment with this agent's action and a fixed action for the other agent"""
        # Get action from fixed policy for the other agent
        if self.fixed_policy is not None:
            other_obs = self.latest_obs[self.other_agent_key]
            other_action, _ = self.fixed_policy.predict(other_obs, deterministic=False)
        else:
            # Random action if no fixed policy
            other_action = self.env.action_space[self.other_agent_key].sample()

        # Combine actions
        actions = {self.agent_key: action, self.other_agent_key: other_action}

        # Step the environment
        self.latest_obs, rewards, terminated, truncated, info = self.env.step(actions)

        # Return only this agent's observation and reward
        return (
            self.latest_obs[self.agent_key],
            rewards[self.agent_key],
            terminated,
            truncated,
            info,
        )


#######################################
# BC Initialization Helpers
#######################################


def initialize_policy_from_bc(policy, bc_model):
    """
    Initialize a policy network from a pretrained BC model

    Parameters:
    policy: Policy to initialize
    bc_model: Pretrained BC model
    """
    with torch.no_grad():
        # Extract linear layers from BC model
        bc_linear_layers = [
            layer for layer in bc_model.fc if isinstance(layer, nn.Linear)
        ]

        # Extract linear layers from policy network
        policy_linear_layers = []
        for layer in policy.mlp_extractor.policy_net:
            if isinstance(layer, nn.Linear):
                policy_linear_layers.append(layer)

        # Copy weights for hidden layers
        for i in range(min(len(bc_linear_layers) - 1, len(policy_linear_layers))):
            policy_linear_layers[i].weight.copy_(bc_linear_layers[i].weight)
            policy_linear_layers[i].bias.copy_(bc_linear_layers[i].bias)

        # Copy output layer
        policy.action_net.weight.copy_(bc_linear_layers[-1].weight)
        policy.action_net.bias.copy_(bc_linear_layers[-1].bias)

        print(
            f"Initialized policy from BC model with {len(bc_linear_layers)} linear layers"
        )


def save_bc_compatible_model(model, output_path, input_size):
    """
    Save a trained policy in a format compatible with BehaviorClonedAgent

    Parameters:
    model: Trained PPO model
    output_path: Path to save the BC-compatible model
    input_size: Input size for the BC model
    """
    # Create a new BC model
    bc_model = BehaviorCloningModel(input_size, 4)

    with torch.no_grad():
        # Get policy networks
        policy = model.policy

        # Extract only the linear layers from the BC model
        bc_layers = [layer for layer in bc_model.fc if isinstance(layer, nn.Linear)]

        # Find all linear layers in the policy network
        policy_layers = []
        for layer in policy.mlp_extractor.policy_net:
            if isinstance(layer, nn.Linear):
                policy_layers.append(layer)

        # Copy weights for all hidden layers
        for i in range(len(bc_layers) - 1):  # Exclude the output layer
            if i < len(policy_layers):
                bc_layers[i].weight.copy_(policy_layers[i].weight)
                bc_layers[i].bias.copy_(policy_layers[i].bias)

        # Copy the output layer separately
        bc_layers[-1].weight.copy_(policy.action_net.weight)
        bc_layers[-1].bias.copy_(policy.action_net.bias)

    # Save the model
    torch.save(bc_model.state_dict(), output_path)
    print(f"Saved BC-compatible model to {output_path}")


#######################################
# Wandb Callback
#######################################


class WandbCallback(BaseCallback):
    """Callback for logging training metrics to wandb"""

    def __init__(self, prefix="agent", verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.prefix = prefix
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self):
        # Log episode stats when episode is done
        if self.locals.get("dones") and any(self.locals["dones"]):
            # Check if episode info is available
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    # Extract episode statistics
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]

                    # Store for calculating averages
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.episode_count += 1

                    # Log to wandb
                    wandb.log(
                        {
                            f"{self.prefix}/episode_reward": ep_reward,
                            f"{self.prefix}/episode_length": ep_length,
                            f"{self.prefix}/avg_reward": np.mean(
                                self.episode_rewards[-100:]
                            ),
                            f"{self.prefix}/avg_length": np.mean(
                                self.episode_lengths[-100:]
                            ),
                            f"{self.prefix}/episodes": self.episode_count,
                        }
                    )
                    break  # Only log once per step

        return True


#######################################
# Simplified Training Function
#######################################


def train_alternating(env, args):
    """
    Train both agents by alternating between them, using model.learn().

    Parameters:
    env: Multi-agent environment (MARLGymEnv)
    args: Command line arguments

    Returns:
    model_p1, model_p2: Trained models for both agents
    """
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))

    # Load BC models for initialization
    bc_model_p1 = BehaviorCloningModel(191, 4)  # Using known input size
    bc_model_p1.load_state_dict(torch.load(args.p1_model, weights_only=True))

    bc_model_p2 = BehaviorCloningModel(191, 4)  # Using known input size
    bc_model_p2.load_state_dict(torch.load(args.p2_model, weights_only=True))

    # Define policy architecture
    policy_kwargs = {
        "net_arch": [dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])],
        "activation_fn": nn.ReLU,
    }

    # Create environment wrappers for each agent
    env_p1 = AlternatingAgentWrapper(env, 1)
    env_p2 = AlternatingAgentWrapper(env, 2)

    # Create PPO models for each agent
    model_p1 = PPO(
        "MlpPolicy",
        env_p1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    model_p2 = PPO(
        "MlpPolicy",
        env_p2,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # Initialize from BC models
    initialize_policy_from_bc(model_p1.policy, bc_model_p1)
    initialize_policy_from_bc(model_p2.policy, bc_model_p2)

    # CRITICAL: Set the fixed policies to ensure both agents are moving from the start
    env_p1.set_fixed_policy(model_p2.policy)
    env_p2.set_fixed_policy(model_p1.policy)

    # Create callbacks for logging
    callback_p1 = WandbCallback(prefix="agent1")
    callback_p2 = WandbCallback(prefix="agent2")

    # Train by alternating
    iterations = 10
    steps_per_iteration = args.timesteps // iterations // 2  # Half for each agent

    print("Starting alternating training...")
    print("Both agents will be active in the environment at all times.")
    print("When agent 1 is learning, agent 2 uses its current policy.")
    print("When agent 2 is learning, agent 1 uses its current policy.")

    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")

        # Train agent 1
        print(
            f"Training agent 1 for {steps_per_iteration} steps (agent 2 follows fixed policy)..."
        )
        model_p1.learn(
            total_timesteps=steps_per_iteration,
            callback=callback_p1,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        # Update agent 2's environment with the latest policy from agent 1
        env_p2.set_fixed_policy(model_p1.policy)

        # Reset environment
        env.reset()

        # Train agent 2
        print(
            f"Training agent 2 for {steps_per_iteration} steps (agent 1 follows fixed policy)..."
        )
        model_p2.learn(
            total_timesteps=steps_per_iteration,
            callback=callback_p2,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        # Update agent 1's environment with the latest policy from agent 2
        env_p1.set_fixed_policy(model_p2.policy)

        # Reset environment
        env.reset()

        # Save intermediate models
        if (i + 1) % 2 == 0:
            p1_output_path = os.path.join(
                args.output_dir, f"{args.exp_name}_agent1_iter{i + 1}.pth"
            )
            p2_output_path = os.path.join(
                args.output_dir, f"{args.exp_name}_agent2_iter{i + 1}.pth"
            )
            save_bc_compatible_model(model_p1, p1_output_path, input_size=191)
            save_bc_compatible_model(model_p2, p2_output_path, input_size=191)
            print(f"Saved intermediate models at iteration {i + 1}")

    # Clean up wandb
    if args.use_wandb:
        wandb.finish()

    return model_p1, model_p2


#######################################
# Main Training Loop
#######################################


def create_marl_env(args):
    """Create a multi-agent environment for training"""
    from src.marl_gym_env import MARLGymEnv

    # Create the multi-agent environment
    env = MARLGymEnv(
        level_index=args.level,
        p1_bc_model_path=args.p1_model,
        p2_bc_model_path=args.p2_model,
    )

    return env


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train MARL agents for Chip's Challenge"
    )

    # Environment settings
    parser.add_argument("--level", type=int, default=0, help="Level index to train on")
    parser.add_argument(
        "--p1-model",
        type=str,
        default="./model/lv1_bc_model_3.8.pth",
        help="Path to BC model for player 1",
    )
    parser.add_argument(
        "--p2-model",
        type=str,
        default="./model/lv1_bc_model_3.8.pth",
        help="Path to BC model for player 2",
    )

    # Training settings
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Total timesteps to train"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048, help="Number of steps per update"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for updates"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="Number of epochs per update"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )

    # Output settings
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--use-wandb", default=True, help="Whether to use wandb")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="chips-challenge-marl",
        help="W&B project name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./model", help="Output directory"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Set experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"marl_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    # Create environment
    env = create_marl_env(args)

    try:
        # Train agents with alternating approach
        model_p1, model_p2 = train_alternating(env, args)

        # Save trained models in BC-compatible format
        p1_output_path = os.path.join(
            args.output_dir, f"{args.exp_name}_agent1_final.pth"
        )
        p2_output_path = os.path.join(
            args.output_dir, f"{args.exp_name}_agent2_final.pth"
        )

        save_bc_compatible_model(model_p1, p1_output_path, input_size=191)
        save_bc_compatible_model(model_p2, p2_output_path, input_size=191)

        print("Training complete!")
        print(f"Agent 1 model saved to: {p1_output_path}")
        print(f"Agent 2 model saved to: {p2_output_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up
        env.close()


if __name__ == "__main__":
    main()
