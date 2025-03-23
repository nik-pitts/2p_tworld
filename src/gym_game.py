import os
import pygame
import src.settings as settings
from src.tiles import TileSpriteSheet, TileWorld
from src.agent import BehaviorClonedAgent, RLAgent
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
        self.max_steps = 200  # Maximum steps before ending episode

        self.action_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

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
            is_train=False,
        )

        # Create player 2 - RL Agent that will learn from scratch
        self.player2 = RLAgent(
            player_positions[1][0],
            player_positions[1][1],
            self.tile_world,
            self,
            2,
            is_train=True,
        )

        # Initialize player2 training variables
        self.player2.prev_collected_chips = 0
        self.player2.max_distance_from_start = 0
        self.player2.prev_position = (self.player2.x, self.player2.y)
        self.player2.stuck_count = 0
        self.player2.visited_positions = set()

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
        prev_chips_collected = self.player2.collected_chips
        prev_socket_unlocked = self.tile_world.socket_unlocked
        prev_player2_alive = self.player2.alive

        # Process player 1 action (from fixed BC model)
        # Use predicted action
        p1_action = self.player1.predict_action()
        self.player1.step(p1_action)

        # Process player 2 action (from RL model)
        # Get action from internal model
        p2_action = action
        # Execute the action
        self.player2.step(self.action_mapping[p2_action])

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
            prev_player2_alive,
            level_complete,
        )

        # Check if episode is done
        terminated = level_complete or game_over or self.steps >= self.max_steps

        truncated = self.steps >= self.max_steps

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
            "p1_chips_collected": self.player1.collected_chips,
            "p2_chips_collected": self.player2.collected_chips,
            "chips_collected": self.tile_world.collected_chips,
            "socket_unlocked": self.tile_world.socket_unlocked,
            "p2_action": p2_action
            if isinstance(self.player2, BehaviorClonedAgent)
            else None,
        }

        return observation, reward, terminated, truncated, info

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

    def _calculate_reward(self, prev_chips, prev_socket, prev_alive, level_complete):
        """
        Reward function focused on efficient level completion:
        1. Collect chips (necessary)
        2. Unlock sockets (necessary)
        3. Complete level as quickly as possible

        Parameters are the same as before
        """
        reward = 0

        # Get current state
        curr_chips = self.player2.collected_chips
        curr_socket = self.tile_world.socket_unlocked
        curr_alive = self.player2.alive
        curr_pos = (self.player2.x, self.player2.y)

        # Track if agent is stuck (oscillating between positions)
        if not hasattr(self.player2, "prev_position"):
            self.player2.prev_position = curr_pos
            self.player2.stuck_count = 0

        if self.player2.prev_position == curr_pos:
            self.player2.stuck_count += 1
        else:
            self.player2.stuck_count = 0

        # Update previous position
        self.player2.prev_position = curr_pos

        # ----- MAJOR REWARD COMPONENTS -----

        # 1. Chip Collection - Essential task
        if curr_chips > prev_chips:
            # Reward for collecting a chip (higher value to emphasize importance)
            reward += 30
            self.prev_collected_chips = curr_chips
            print(
                f"Totoal: {self.tile_world.collected_chips}, Prev:{prev_chips}, Curr:{curr_chips} Collected!"
            )

        # 2. Socket Unlock - Critical milestone
        if curr_socket and not prev_socket:
            # Big reward for unlocking socket
            reward += 50.0
            print("Socket Unlocked!")

        # 3. Level Completion - Primary goal with strong efficiency incentive
        if level_complete:
            # Base reward for completion
            completion_reward = 100.0

            # Apply significant step penalty for efficiency
            # This creates a curve where faster completions get much higher rewards
            # For example:
            # - Complete in 100 steps: reward = 100 * (1.0 - (100/1000)^0.5) = 68.4
            # - Complete in 500 steps: reward = 100 * (1.0 - (500/1000)^0.5) = 29.3
            efficiency_factor = (
                self.steps / self.max_steps
            ) ** 0.5  # Square root for gentler curve
            time_bonus = completion_reward * (1.0 - efficiency_factor)

            reward += time_bonus

            print(
                f"Level Complete in {self.steps} steps! Efficiency bonus: {time_bonus:.1f}"
            )

        # ----- PENALTIES -----

        # 1. Step penalty - INCREASED to discourage long episodes
        # This is critical for encouraging efficiency
        step_penalty = 0.1  # Increased from 0.02
        reward -= step_penalty

        # 2. Stuck penalty - Discourage oscillating in place
        if self.player2.stuck_count > 3:
            stuck_penalty = 0.2 * self.player2.stuck_count  # Increased multiplier
            reward -= min(2.0, stuck_penalty)  # Increased cap

        # 3. Death penalty - Severe to avoid deaths
        if not curr_alive and prev_alive:
            # print("Player 2 died!")
            reward -= 20.0  # Increased from 10.0

        # ----- MINOR REWARD COMPONENTS -----

        # Only reward movement if it's toward an unexplored area or objective
        # Instead of rewarding distance from start
        # Check if position is new
        pos_tuple = curr_pos
        if pos_tuple not in self.player2.visited_positions:
            # Small reward for exploring new positions
            reward += 0.5
            self.player2.visited_positions.add(pos_tuple)

        return reward

    def _get_observation(self):
        """
        Get observation vector for RL agent
        This creates a vector representation of the game state
        Includes both player1's state and knowledge about player2
        """
        # Get player1's state vector (includes knowledge of player2)
        state_vector = self.player2.get_state_vector().squeeze(0).detach().numpy()

        # Concatenate to create full observation
        return state_vector

    def reset(self):
        """Reset the game to initial state and return initial observation"""
        self.load_game()
        return self._get_observation()

    def close(self):
        """Clean up resources"""
        pygame.quit()
