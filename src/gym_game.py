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
        self.max_steps = 1000  # Maximum steps before ending episode

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

        # P1 related varaible for training
        self.player1.prev_collected_chips = 0
        self.player1.max_distance_from_start = 0
        self.player1.prev_position = (self.player1.x, self.player1.y)
        self.player1.stuck_count = 0

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

        # Process player 1 action (from RL policy)
        # Use predicted action
        self.player1.step(self.action_mapping[action])

        # Process player 2 action (from fixed BC model)
        # Get action from internal model
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
            "p1_chips_collected": self.player1.collected_chips,
            "p2_chips_collected": self.player2.collected_chips,
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

    def _calculate_reward(self, prev_chips, prev_socket, prev_p1_alive, level_complete):
        """Calculate reward based on state changes with emphasis on active cooperation"""
        reward = 0

        # Reward for collecting chips specifically by player1 (RL agent)
        if self.player1.collected_chips > self.player1.prev_collected_chips:
            chips_collected_by_p1 = (
                self.player1.collected_chips - self.player1.prev_collected_chips
            )
            reward += 2.0 * chips_collected_by_p1

        # Smaller reward for teammate collecting chips (still want cooperative behavior)
        if self.tile_world.collected_chips > prev_chips:
            reward += 0.5

        # Reward for unlocking socket - higher reward if player1 collected critical chips
        if self.tile_world.socket_unlocked and not prev_socket:
            # Check if player1 collected at least half the chips
            if self.player1.collected_chips >= self.tile_world.total_chips / 2:
                reward += 8.0  # Higher reward for significant contribution
            else:
                reward += 3.0  # Lower reward if mostly carried by player2

        # Movement and exploration rewards
        # Distance from starting position
        start_pos = self.tile_world.player_positions[0]
        current_pos = (self.player1.x, self.player1.y)
        manhattan_distance = abs(current_pos[0] - start_pos[0]) + abs(
            current_pos[1] - start_pos[1]
        )

        # Small reward for exploration
        if manhattan_distance > self.player1.max_distance_from_start:
            self.player1.max_distance_from_start = manhattan_distance
            reward += 0.2  # Reward for exploring new areas

        # Proximity reward for getting closer to objectives
        if not self.tile_world.socket_unlocked:
            # Reward for getting closer to chips if socket isn't unlocked
            nearest_chip = self.tile_world.find_nearest_chip(
                self.player1.x, self.player1.y, "CHIP"
            )
            if nearest_chip != (-1, -1):  # If there are chips left
                if hasattr(self.player1, "prev_chip_distance"):
                    prev_distance = self.player1.prev_chip_distance
                    current_distance = abs(self.player1.x - nearest_chip[0]) + abs(
                        self.player1.y - nearest_chip[1]
                    )
                    if current_distance < prev_distance:
                        reward += 0.1  # Small reward for moving toward chips
                    self.player1.prev_chip_distance = current_distance
                else:
                    # Initialize on first call
                    self.player1.prev_chip_distance = abs(
                        self.player1.x - nearest_chip[0]
                    ) + abs(self.player1.y - nearest_chip[1])
        else:
            # If socket is unlocked, reward for moving toward exit
            exit_pos = self.tile_world.exit_position
            if hasattr(self.player1, "prev_exit_distance"):
                prev_distance = self.player1.prev_exit_distance
                current_distance = abs(self.player1.x - exit_pos[0]) + abs(
                    self.player1.y - exit_pos[1]
                )
                if current_distance < prev_distance:
                    reward += 0.15  # Reward for moving toward exit
                self.player1.prev_exit_distance = current_distance
            else:
                self.player1.prev_exit_distance = abs(
                    self.player1.x - exit_pos[0]
                ) + abs(self.player1.y - exit_pos[1])

        # Reward for level completion - scale by player1's contribution
        if level_complete:
            # Base completion reward
            completion_reward = 15.0

            # Bonus for player1 actively contributing (exiting or collecting chips)
            if self.player1.exited:
                reward += completion_reward * 10  # Highest reward if player1 exits
            elif self.player1.collected_chips > 0:
                # Scale reward by proportion of chips collected
                contribution_ratio = self.player1.collected_chips / max(
                    1, self.tile_world.total_chips
                )
                reward += completion_reward * (0.5 + contribution_ratio)
            else:
                # Minimal reward for completion without contribution
                reward += completion_reward * 0.3

        # Penalty for player death
        if prev_p1_alive and not self.player1.alive:
            reward -= 10.0  # Severe penalty for dying

        # Penalty for getting stuck or not moving
        if hasattr(self.player1, "prev_position"):
            if self.player1.prev_position == (self.player1.x, self.player1.y):
                self.player1.stuck_count += 1
                if self.player1.stuck_count > 3:  # If stuck for multiple steps
                    reward -= 10 * self.player1.stuck_count  # Increasing penalty
            else:
                self.player1.stuck_count = 0

        # Update previous position
        self.player1.prev_position = (self.player1.x, self.player1.y)
        self.player1.prev_collected_chips = self.player1.collected_chips

        # Time pressure - increasing penalty as steps increase
        time_penalty = min(0.05, 0.01 * (self.steps / 30))  # Gradually increases
        reward -= time_penalty

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
