import os
import pygame
import src.settings as settings
from src.tiles import TileSpriteSheet, TileWorld
from src.agent import BehaviorClonedAgent
import numpy as np


class GymGame:
    """
    Game class for RL Gym training with two agents:
    - Player 1: BC agent being trained with RL
    - Player 2: Fixed BC agent that doesn't change
    """

    def __init__(self, level_index=0, p1_bc_model_path=None, p2_bc_model_path=None):
        # Set environment variables for headless mode
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

        # Initialize pygame in headless mode
        pygame.init()

        # Create a minimal surface
        self.screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))

        # Load tile sprite sheet
        self.tile_sprite_sheet = TileSpriteSheet(
            settings.TILE_SHEET_PATH, settings.TILE_SIZE
        )

        # Load tile world
        self.tile_world = TileWorld(settings.LEVEL_DATA_PATH, self.tile_sprite_sheet)
        self.tile_world.level_index = level_index

        # BC model paths for both agents
        self.p1_bc_model_path = p1_bc_model_path
        self.p2_bc_model_path = p2_bc_model_path

        # Track steps for the current episode
        self.steps = 0
        self.max_steps = 500  # Maximum steps before ending episode

    def load_game(self):
        """Initialize or reset the game to initial state"""
        # Load level
        self.tile_world.load_level(self.tile_world.level_index)

        # Get player positions
        player_positions = self.tile_world.player_positions
        if len(player_positions) < 2:
            raise ValueError(
                "At least two player positions must be defined in the level data!"
            )

        # Create player 1 - BC agent that will be trained with RL
        self.player1 = BehaviorClonedAgent(
            player_positions[0][0],
            player_positions[0][1],
            self.tile_world,
            self,
            1,
            self.p1_bc_model_path,
            is_train=True,
        )

        # Create player 2 - Fixed BC agent
        self.player2 = BehaviorClonedAgent(
            player_positions[1][0],
            player_positions[1][1],
            self.tile_world,
            self,
            2,
            self.p2_bc_model_path,
            is_train=True,
        )

        # Set up the minimal UI for tile_world to function
        # Some game mechanics reference UI
        dummy_ui = type(
            "DummyUI",
            (),
            {
                "update_inventory": lambda *args: None,
                "clear_hint": lambda: None,
                "show_hint": lambda *args: None,
            },
        )()
        self.tile_world.game_ui = dummy_ui

        # Reset game state
        self.steps = 0

    def step(self, action):
        """
        Take a single step in the game environment

        Parameters:
        action: Integer 0-3 representing player1's action (UP, DOWN, LEFT, RIGHT)
                Player2's action is determined by its BC model

        Returns:
        observation, reward, done, info
        """
        self.steps += 1

        # Save previous state for reward calculation
        prev_chips_collected = self.tile_world.collected_chips
        prev_socket_unlocked = self.tile_world.socket_unlocked
        prev_player1_alive = self.player1.alive
        prev_player2_alive = self.player2.alive

        # Process player 1 action (from RL policy)
        # Direct execution for BC agent
        self.player1.step(action)

        # Process player 2 action (from fixed BC model)
        # Get action from fixed BC model
        p2_action = self.player2.predict_action()
        # Execute the action
        self.player2.step(p2_action)

        # Only applies to level 1
        if self.tile_world.level_index == 1:
            # Process player 1 animations
            self.player1.update_forced_movement()
            self.player1.update_sliding_movement()

            # Process player 2 animations
            self.player2.update_forced_movement()
            self.player2.update_sliding_movement()

            # Check collisions
            if self.player1.collision_detection(self.player1.x, self.player1.y):
                self.player1.remove_self()

            if self.player2.collision_detection(self.player2.x, self.player2.y):
                self.player2.remove_self()

            # Move beetles/monsters
            for beetle in self.tile_world.beetles:
                beetle.move()

        # Check for level completion or game over
        level_complete = self.check_level_complete()
        game_over = self.check_game_over()

        # Calculate reward
        reward = self._calculate_reward(
            prev_chips_collected,
            prev_socket_unlocked,
            prev_player1_alive,
            prev_player2_alive,
            level_complete,
        )

        # Check if episode is done
        done = level_complete or game_over or self.steps >= self.max_steps

        # Get observation
        observation = self._get_observation()

        # Additional info
        info = {
            "level_complete": level_complete,
            "game_over": game_over,
            "steps": self.steps,
            "player1_alive": self.player1.alive,
            "player2_alive": self.player2.alive,
            "player1_pos": (self.player1.x, self.player1.y),
            "player2_pos": (self.player2.x, self.player2.y),
            "chips_collected": self.tile_world.collected_chips,
            "socket_unlocked": self.tile_world.socket_unlocked,
            "p2_action": p2_action
            if isinstance(self.player2, BehaviorClonedAgent)
            else None,
        }

        return observation, reward, done, info

    def check_level_complete(self):
        """Checks if either player has reached the exit."""
        if self.player1.exited or self.player2.exited:
            return True
        return False

    def check_game_over(self):
        """Checks if both players are dead."""
        if not (self.player1.alive and self.player2.alive):
            return True
        return False

    def _calculate_reward(
        self, prev_chips, prev_socket, prev_p1_alive, prev_p2_alive, level_complete
    ):
        """Calculate reward based on state changes"""
        reward = 0

        # Reward for collecting chips
        if self.tile_world.collected_chips > prev_chips:
            reward += 1.0

        # Reward for unlocking socket
        if self.tile_world.socket_unlocked and not prev_socket:
            reward += 5.0

        # Reward for level completion
        if level_complete:
            reward += 20.0

        # Penalty for player death
        if prev_p1_alive and not self.player1.alive:
            reward -= 5.0

        if prev_p2_alive and not self.player2.alive:
            reward -= 5.0

        # Small step penalty to encourage efficiency
        reward -= 0.01

        return reward

    def _get_observation(self):
        """
        Get observation vector for RL agent
        This creates a vector representation of the game state
        Includes both player1's state and knowledge about player2
        """
        # Get player1's state vector (includes knowledge of player2)
        state_vector = self.player1.get_state_vector().squeeze(0).detach().numpy()

        # Add additional information about player2
        # p2_pos = np.array([self.player2.x, self.player2.y])
        p2_collected = np.array([self.player2.collected_chips])  # 1
        p2_alive = np.array([float(self.player2.alive)])  # 1

        # Concatenate to create full observation
        return np.concatenate([state_vector, p2_collected, p2_alive])

    def reset(self):
        """Reset the game to initial state and return initial observation"""
        self.load_game()
        return self._get_observation()

    def close(self):
        """Clean up resources"""
        pygame.quit()
