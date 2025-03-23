#!/usr/bin/env python3
"""
Modified Multi-Agent Reinforcement Learning (MARL) training script for Chip's Challenge
Trains a single agent (Agent 2) from scratch, while Agent 1 uses a fixed BC policy.
"""

import os
import numpy as np
from tabulate import tabulate
import torch.nn as nn
import wandb
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback


#######################################
# Agent Wrapper Classes
#######################################


class BCRLWrapper(gym.Wrapper):
    """
    Wrapper for training Agent 2 with a fixed BC policy for Agent 1.
    Agent 1 always uses its BC model, while Agent 2 learns from scratch.
    """

    def __init__(self, env):
        super().__init__(env)
        # We're focusing on training agent 2
        self.agent_key = "agent2"
        self.other_agent_key = "agent1"

        # Store the latest observation for both agents
        self.latest_obs = None

        # Define observation and action spaces for the learning agent (agent 2)
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space[self.agent_key]
            self.action_space = env.action_space[self.agent_key]
        else:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        self.game = env.game

    def reset(self, **kwargs):
        """Reset the environment and return agent 2's observation"""
        self.latest_obs, info = self.env.reset(**kwargs)
        return self.latest_obs, info

    def step(self, action):
        """
        Step the environment with agent 2's action and BC action for agent 1
        Agent 1 uses its BC model (handled by the environment)
        """
        # Prepare actions dictionary - agent 1's action will be determined by BC model

        # The environment will automatically use the BC model for agent 1
        self.latest_obs, rewards, terminated, truncated, info = self.env.step(action)

        # Return only agent 2's observation and reward
        return (
            self.latest_obs,
            rewards,
            terminated,
            truncated,
            info,
        )


#######################################
# Wandb Callback
#######################################


class WandbCallback(BaseCallback):
    """Callback for logging training metrics to wandb"""

    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
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
                            "episode_reward": ep_reward,
                            "episode_length": ep_length,
                            "avg_reward": np.mean(self.episode_rewards[-100:]),
                            "avg_length": np.mean(self.episode_lengths[-100:]),
                            "episodes": self.episode_count,
                        }
                    )
                    break  # Only log once per step

        return True


#######################################
# Main Training Function
#######################################


def train_rl_agent(env, config, model=None):
    """
    Train agent 2 from scratch or continue training an existing model.

    Parameters:
    env: Environment with BC agent 1
    config: Configuration dictionary with training parameters
    model: Optional existing model to continue training

    Returns:
    model: Trained model for agent 2
    """
    # Initialize wandb
    if config["use_wandb"]:
        wandb.login(key="294ac5de6babc54da53b9aadb344b3bb173b314d")
        wandb.init(
            project=config["wandb_project"], name=config["exp_name"], config=config
        )
        wandb.config.update(config)

    # Create environment wrapper
    wrapped_env = BCRLWrapper(env)

    # If we don't have a model, create a new one
    if model is None:
        print("Starting training of agent 2 from scratch...")

        # Define policy architecture
        policy_kwargs = {
            "net_arch": dict(pi=[64, 64, 64], vf=[64, 64, 64]),
            "activation_fn": nn.ReLU,
        }

        # Create PPO model for agent 2
        model = PPO(
            "MlpPolicy",
            wrapped_env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    else:
        print("Continuing training from existing model...")
        # Update the model's environment
        model.set_env(wrapped_env)

    # Create wandb callback
    callback = WandbCallback()

    # Train agent 2
    model.learn(
        total_timesteps=config["timesteps"],
        callback=callback,
        reset_num_timesteps=model is None,  # Only reset if new training
        progress_bar=True,
    )

    # Clean up wandb
    if config["use_wandb"]:
        wandb.finish()

    return model


#######################################
# Environment Creation
#######################################


def create_rl_env(config):
    """Create a multi-agent environment for training"""
    from src.gym_env import GymEnv

    # Create the multi-agent environment
    # Note: We set train_agent_id=2 to focus on agent 2 and use BC for agent 1
    env = GymEnv(
        level_index=config["level"],
        p1_bc_model_path=config["p1_model"],  # BC model for agent 1
        p2_bc_model_path=None,  # No BC model for agent 2 - learning from scratch
    )

    return env


def get_config():
    """Get training configuration"""
    from datetime import datetime

    # Create config dictionary with all parameters
    config = {
        # Environment settings
        "level": 0,  # Level index to train on
        "p1_model": "./model/lv1_bc_model_3.8.pth",  # Path to BC model for player 1
        # Training settings
        "timesteps": 100000,  # Total timesteps to train
        "seed": 42,  # Random seed
        "learning_rate": 3e-4,  # Learning rate
        "n_steps": 2048,  # Number of steps per update
        "batch_size": 128,  # Batch size for updates
        "n_epochs": 15,  # Number of epochs per update
        "gamma": 0.995,  # Discount factor
        "clip_range": 0.2,  # PPO clip range
        "ent_coef": 0.005,  # Entropy coefficient
        # Output settings
        "exp_name": f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Experiment name
        "use_wandb": True,  # Whether to use wandb
        "wandb_project": "chips-challenge-rl",  # W&B project name
        "output_dir": "./model",  # Output directory
    }

    return config


def main():
    """Main training function with continue option"""
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description="Train RL agent for Chip's Challenge")
    parser.add_argument("--inspect", action="store_true", help="Run in inspection mode")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes for inspection"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Steps per episode for inspection"
    )
    parser.add_argument(
        "--continue_from", type=str, help="Path to model to continue training from"
    )
    args = parser.parse_args()

    # Get configuration parameters
    config = get_config()

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Set random seed
    set_random_seed(config["seed"])

    # If in inspection mode, run the inspection and exit
    if args.inspect:
        print("Running in inspection mode...")
        inspect_environment(
            config, num_episodes=args.episodes, steps_per_episode=args.steps
        )
        return

    # Create environment for training
    env = create_rl_env(config)
    wrapped_env = BCRLWrapper(env)

    try:
        # Check if continuing from a saved model
        if args.continue_from:
            print(f"Continuing training from model: {args.continue_from}")

            # Load the saved model
            model = PPO.load(args.continue_from, env=wrapped_env)

            # Create a new experiment name for continued training
            config["exp_name"] = (
                f"{os.path.splitext(os.path.basename(args.continue_from))[0]}_continued"
            )
        else:
            # Train agent 2 from scratch
            model = None

        # Train (or continue training) the model
        model = train_rl_agent(env, config, model=model)

        # Save trained model
        output_path = os.path.join(
            config["output_dir"], f"{config['exp_name']}_agent2_final.pth"
        )
        model.save(output_path)

        print("Training complete!")
        print(f"Agent 2 model saved to: {output_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save model even if interrupted
        if "model" in locals():
            interrupted_path = os.path.join(
                config["output_dir"], f"{config['exp_name']}_interrupted.pth"
            )
            model.save(interrupted_path)
            print(f"Interrupted model saved to: {interrupted_path}")
    finally:
        # Clean up
        env.close()


def inspect_environment(config, num_episodes=1, steps_per_episode=20):
    """
    Inspect the environment to verify observation and action spaces,
    and visualize sample trajectories.

    Parameters:
    config: Configuration dictionary
    num_episodes: Number of episodes to run for inspection
    steps_per_episode: Steps per episode to run
    """

    # Create environment
    env = create_rl_env(config)
    wrapped_env = BCRLWrapper(env)

    print("\n" + "=" * 80)
    print("ENVIRONMENT INSPECTION")
    print("=" * 80)

    # Inspect observation space
    print("\nObservation Space:")
    print(f"  Type: {type(wrapped_env.observation_space)}")
    print(f"  Shape: {wrapped_env.observation_space.shape}")
    print(f"  Low: {wrapped_env.observation_space.low}")
    print(f"  High: {wrapped_env.observation_space.high}")

    # Inspect action space
    print("\nAction Space:")
    print(f"  Type: {type(wrapped_env.action_space)}")
    print(f"  n: {wrapped_env.action_space.n}")
    print(f"  Action mapping: {wrapped_env.game.action_mapping}")

    # Run a few episodes to see actual observations and rewards
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}:")
        obs, _ = wrapped_env.reset()

        # Store episode data for analysis
        episode_data = []

        for step in range(steps_per_episode):
            # Take random action
            action = wrapped_env.action_space.sample()

            # Step environment
            next_obs, reward, terminated, truncated, info = wrapped_env.step(action)

            # Store step data
            step_data = {
                "step": step,
                "action": action,
                "action_name": wrapped_env.game.action_mapping[action],
                "reward": reward,
                "player1_pos": info.get("player1_pos", None),
                "player2_pos": info.get("player2_pos", None),
                "chips_collected": info.get("chips_collected", 0),
                "done": terminated or truncated,
                "p2_action": info.get("p2_action", None),
            }

            episode_data.append(step_data)

            # Print current state
            if step % 5 == 0:  # Print every 5 steps to avoid clutter
                print(f"\n  Step {step}:")
                print(
                    f"    Action: {action} ({wrapped_env.game.action_mapping[action]})"
                )
                print(f"    Reward: {reward}")
                print(f"    Player1 pos: {info.get('player1_pos', None)}")
                print(f"    Player2 pos: {info.get('player2_pos', None)}")
                print(f"    Chips collected: {info.get('chips_collected', 0)}")
                print(f"    Socket unlocked: {info.get('socket_unlocked', False)}")

            # Break if done
            if terminated or truncated:
                print(f"  Episode ended after {step + 1} steps")
                break

            # Update observation
            obs = next_obs

        # Print episode summary
        print("\n  Episode Summary:")
        print("    Step | Action | Reward | P1 Pos | P2 Pos | Chips")
        print("    " + "-" * 60)

        # Convert episode data to table for better visualization
        table_data = []
        for entry in episode_data:
            table_data.append(
                [
                    entry["step"],
                    f"{entry['action']} ({entry['action_name']})",
                    f"{entry['reward']:.3f}",
                    entry["player1_pos"],
                    entry["player2_pos"],
                    entry["chips_collected"],
                ]
            )

        print(
            tabulate(
                table_data,
                headers=["Step", "Action", "Reward", "P1 Pos", "P2 Pos", "Chips"],
            )
        )

        # Analyze observation
        print("\n  Observation Analysis:")
        if isinstance(obs, np.ndarray):
            print(f"    Shape: {obs.shape}")
            print(
                f"    Min: {obs.min():.3f}, Max: {obs.max():.3f}, Mean: {obs.mean():.3f}"
            )
            print(f"    Non-zero elements: {np.count_nonzero(obs)}/{obs.size}")

            # Sample of the observation (first 10 elements)
            print(f"    Sample (first 10 elements): {obs.flatten()[:10]}")

    # Close environment
    wrapped_env.close()

    print("\n" + "=" * 80)
    print("ENVIRONMENT INSPECTION COMPLETE")
    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    main()
