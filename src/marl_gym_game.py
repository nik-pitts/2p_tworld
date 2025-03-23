import os
import pygame
import numpy as np
import torch
import src.settings as settings
from src.tiles import TileSpriteSheet, TileWorld
from src.agent import BehaviorClonedAgent


class MARLGymGame:
    """
    Game class for Multi-Agent RL (MARL) training that leverages pretrained BC models.
    - Both agents are based on pretrained BC models
    - Both receive separate observations and rewards
    - Both are trained to cooperate to complete levels faster
    - Maintains compatible architecture with original BehaviorClonedAgent
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
        self.max_steps = 100  # Maximum steps before ending episode

        # Action mapping for both agents
        self.action_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        # Load game initially
        self.load_game()

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

        # Create player 2 - BC agent that will be trained with RL
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

        # Initialize tracking variables for player 1
        self.player1.prev_collected_chips = 0
        self.player1.prev_position = (self.player1.x, self.player1.y)
        self.player1.stuck_count = 0

        # Initialize tracking variables for player 2
        self.player2.prev_collected_chips = 0
        self.player2.prev_position = (self.player2.x, self.player2.y)
        self.player2.stuck_count = 0

        # Reset step counter
        self.steps = 0

    def step_multi_agent(self, p1_action, p2_action):
        """
        Process a single step with actions from both agents simultaneously

        Parameters:
        p1_action: Integer 0-3 representing player1's action
        p2_action: Integer 0-3 representing player2's action

        Returns:
        observations, rewards, done, info
        """
        self.steps += 1

        # Save previous state for reward calculation
        prev_chips_collected = self.tile_world.collected_chips
        prev_socket_unlocked = self.tile_world.socket_unlocked
        # Process player 1 action
        self.player1.step(self.action_mapping[p1_action.item()])

        # Process player 2 action
        self.player2.step(self.action_mapping[p2_action.item()])

        # Process animations, collisions, beetles for both players
        self._process_game_mechanics()

        # Check for level completion or game over
        level_complete = self.check_level_complete()
        game_over = self.check_game_over()

        # Calculate separate rewards for each agent
        p1_reward = self._calculate_agent_reward(
            agent=self.player1,
            prev_chips_collected=prev_chips_collected,
            prev_socket_unlocked=prev_socket_unlocked,
            level_complete=level_complete,
        )

        p2_reward = self._calculate_agent_reward(
            agent=self.player2,
            prev_chips_collected=prev_chips_collected,
            prev_socket_unlocked=prev_socket_unlocked,
            level_complete=level_complete,
        )

        # Check if episode is done
        done = level_complete or game_over or self.steps >= self.max_steps

        # Get observations for each agent
        p1_observation = self._get_agent_observation(for_player=self.player1)
        p2_observation = self._get_agent_observation(for_player=self.player2)

        # Create dictionary of observations and rewards
        observations = {"agent1": p1_observation, "agent2": p2_observation}

        rewards = {"agent1": p1_reward, "agent2": p2_reward}

        # Additional info
        info = {
            "level_complete": level_complete,
            "game_over": game_over,
            "steps": self.steps,
            "player1": {
                "alive": self.player1.alive,
                "position": (self.player1.x, self.player1.y),
                "chips_collected": self.player1.collected_chips,
                "exited": self.player1.exited,
            },
            "player2": {
                "alive": self.player2.alive,
                "position": (self.player2.x, self.player2.y),
                "chips_collected": self.player2.collected_chips,
                "exited": self.player2.exited,
            },
            "total_chips_collected": self.tile_world.collected_chips,
            "socket_unlocked": self.tile_world.socket_unlocked,
        }

        return observations, rewards, done, info

    def _process_game_mechanics(self):
        """Process animations, movement physics, and monster logic"""
        # Process player 1 animations and physics
        self.player1.update_forced_movement()
        self.player1.update_sliding_movement()

        # Process player 2 animations and physics
        self.player2.update_forced_movement()
        self.player2.update_sliding_movement()

        # Check collisions with monsters
        if self.player1.collision_detection(self.player1.x, self.player1.y):
            self.player1.remove_self()

        if self.player2.collision_detection(self.player2.x, self.player2.y):
            self.player2.remove_self()

        # Move beetles/monsters
        for beetle in self.tile_world.beetles:
            beetle.move()

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

    def _calculate_agent_reward(
        self,
        agent,
        prev_chips_collected,
        prev_socket_unlocked,
        level_complete,
    ):
        """
        Calculate reward for a specific agent with emphasis on cooperation and speed

        Parameters:
        agent: The agent to calculate reward for (player1 or player2)
        prev_chips_collected: Total chips collected before this step
        prev_socket_unlocked: Whether socket was unlocked before this step
        prev_alive: Whether the agent was alive before this step
        level_complete: Whether the level was completed this step
        prev_steps: Step count before this step (for speed rewards)

        Returns:
        float: The calculated reward
        """
        reward = 0

        # Individual progress rewards
        # -------------------------
        # Reward for agent collecting chips
        if agent.collected_chips > agent.prev_collected_chips:
            chips_collected = agent.collected_chips - agent.prev_collected_chips
            reward += 1.0 * chips_collected  # Direct reward for chip collection

        # Team progress rewards
        # -------------------------
        # Team chip collection (cooperative incentive)
        if self.tile_world.collected_chips > prev_chips_collected:
            reward += 0.5

        # Socket unlocking (milestone)
        if self.tile_world.socket_unlocked and not prev_socket_unlocked:
            # Higher reward for fast unlocking
            steps_factor = 2 * (1.0 - (self.steps / self.max_steps))
            reward += 3.0 * steps_factor  # Reward more for unlocking faster

        # Level completion rewards
        # -------------------------
        if level_complete:
            # Calculate speed bonus - higher reward for faster completion
            # This strongly encourages cooperative speed
            steps_remaining = self.max_steps - self.steps
            speed_factor = steps_remaining / self.max_steps  # 0 to 1
            speed_bonus = 15.0 * (0.5 + speed_factor)  # Base 7.5, max 15.0

            if agent.exited:
                # Higher reward for personally exiting
                reward += speed_bonus * 1.5
            else:
                # Reward for team success even if this agent didn't exit
                # This prevents agents from being selfish
                contribution = max(
                    0.1, agent.collected_chips / self.tile_world.total_chips
                )
                reward += speed_bonus * contribution

        return reward

    def _get_agent_observation(self, for_player):
        """
        Get observation vector for a specific agent

        Parameters:
        for_player: The agent to get observation for

        Returns:
        numpy.ndarray: The observation vector
        """
        # Get base state vector from the agent's perspective
        state_vector = for_player.get_state_vector().detach().numpy()

        # Concatenate to create complete observation
        return state_vector

    def get_bc_actions(self, observations):
        """
        Get actions from pretrained BC models for both agents
        Used for BC regularization during training

        Parameters:
        observations: Dictionary with observations for both agents

        Returns:
        Dictionary with BC-predicted actions for both agents
        """
        # Convert numpy observations to torch tensors
        p1_obs = torch.FloatTensor(observations["agent1"]).unsqueeze(0)
        p2_obs = torch.FloatTensor(observations["agent2"]).unsqueeze(0)

        # Use BC model to predict actions (logits)
        with torch.no_grad():
            # We need to handle the observation properly - extract just the state vector part
            # that the BC model expects (without the additional info about other agent)
            bc_input_size = self.player1.model.fc[0].in_features

            # Extract just the input part the BC model expects
            p1_bc_input = p1_obs[:, :bc_input_size]
            p2_bc_input = p2_obs[:, :bc_input_size]

            # Get BC model outputs (logits)
            p1_logits = self.player1.model(p1_bc_input)
            p2_logits = self.player2.model(p2_bc_input)

            # Convert to probability distributions
            p1_probs = torch.nn.functional.softmax(p1_logits, dim=1)
            p2_probs = torch.nn.functional.softmax(p2_logits, dim=1)

        return {
            "agent1": p1_probs.squeeze(0).numpy(),
            "agent2": p2_probs.squeeze(0).numpy(),
        }

    def reset(self):
        """Reset the game to initial state and return initial observations for both agents"""
        self.load_game()

        # Get initial observations
        p1_observation = self._get_agent_observation(for_player=self.player1)
        p2_observation = self._get_agent_observation(for_player=self.player2)

        return {"agent1": p1_observation, "agent2": p2_observation}

    def close(self):
        """Clean up resources"""
        pygame.quit()
