import gymnasium as gym
from gymnasium import spaces
from src.gym_game import GymGame
import numpy as np


class GymEnv(gym.Env):
    """Gym environment wrapper for Chips Challenge game"""

    metadata = {"render_modes": [None], "render_fps": 60}

    def __init__(self, level_index=0, p1_bc_model_path=None, p2_bc_model_path=None):
        super().__init__()

        # Create the game with both BC model paths
        self.game = GymGame(
            level_index=level_index,
            p1_bc_model_path=p1_bc_model_path,
            p2_bc_model_path=p2_bc_model_path,
        )

        self.game.load_game()

        # Get initial observation to determine space dimensions
        initial_obs = self.game._get_observation()

        print(f"Initial observation shape: {initial_obs.shape}")

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT

        # Observation space based on observation shape
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=initial_obs.shape,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        """Reset the environment"""
        # First reset the seed
        if seed is not None:
            super().reset(seed=seed)

        # Return the observation and empty info dict to match Gym API
        return self.game.reset(), {}

    def step(self, action):
        """
        Take a step in the environment
        action: Integer 0-4 representing agent2's action
        agent1 uses its own BC policy for actions
        """
        observation, reward, terminated, truncated, info = self.game.step(action)

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources"""
        self.game.close()
