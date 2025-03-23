import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from src.marl_gym_game import MARLGymGame
from src.models import BehaviorCloningModel


class MARLGymEnv(gym.Env):
    """
    Multi-Agent Gym Environment for cooperative training
    Wraps MARLGymGame to provide a gymnasium-compatible interface
    Maintains compatibility with BC model architecture for deployment
    """

    metadata = {"render_modes": [None], "render_fps": 60}

    def __init__(
        self,
        level_index=0,
        p1_bc_model_path=None,
        p2_bc_model_path=None,
        train_agent_id=None,
    ):
        """
        Initialize the MARL environment

        Parameters:
        level_index: Index of the level to load
        p1_bc_model_path: Path to BC model for player 1
        p2_bc_model_path: Path to BC model for player 2
        train_agent_id: If specified, only train this agent (1 or 2 or None for both)
        """
        super().__init__()

        # Which agent(s) to train
        self.train_agent_id = train_agent_id  # None = train both

        # Create the game instance
        self.game = MARLGymGame(
            level_index=level_index,
            p1_bc_model_path=p1_bc_model_path,
            p2_bc_model_path=p2_bc_model_path,
        )

        # Load BC models to get input sizes
        if p1_bc_model_path:
            self.p1_model = BehaviorCloningModel(191, 4)  # Known input size
            self.p1_model.load_state_dict(torch.load(p1_bc_model_path))

        if p2_bc_model_path:
            self.p2_model = BehaviorCloningModel(191, 4)  # Known input size
            self.p2_model.load_state_dict(torch.load(p2_bc_model_path))

        # Get initial state to determine observation dimensions
        initial_obs = self.game.reset()

        # Define the observation spaces using the known input dimension
        self.agent1_observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=initial_obs["agent1"].shape,
            dtype=np.float32,
        )

        self.agent2_observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=initial_obs["agent2"].shape,
            dtype=np.float32,
        )

        # Define action spaces (discrete with 4 actions)
        self.agent1_action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT
        self.agent2_action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT

        # Set action and observation space based on training mode
        if self.train_agent_id == 1:
            self.action_space = self.agent1_action_space
            self.observation_space = self.agent1_observation_space
        elif self.train_agent_id == 2:
            self.action_space = self.agent2_action_space
            self.observation_space = self.agent2_observation_space
        else:
            # Multi-agent training - use dictionary spaces
            self.action_space = spaces.Dict(
                {"agent1": self.agent1_action_space, "agent2": self.agent2_action_space}
            )
            self.observation_space = spaces.Dict(
                {
                    "agent1": self.agent1_observation_space,
                    "agent2": self.agent2_observation_space,
                }
            )

    def reset(self, *, seed=None, options=None):
        """Reset the environment and return initial observations"""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        observations = self.game.reset()
        info = {}  # Empty info dict

        # Return appropriate observations based on training mode
        if self.train_agent_id == 1:
            return observations["agent1"], info
        elif self.train_agent_id == 2:
            return observations["agent2"], info
        else:
            return observations, info

    def step(self, action):
        """
        Take a step in the environment

        Parameters:
        action: Either a single action (if training one agent) or
                a dictionary of actions for both agents

        Returns:
        observation, reward, terminated, truncated, info
        """
        # Handle different action formats based on training mode
        if self.train_agent_id == 1:
            # Training only agent 1
            # Get agent 2's action from its BC model
            p1_action = action
            # Use BC model to get action for agent 2
            p2_observation = self._get_current_observation("agent2")
            p2_action = self._get_bc_action(2, p2_observation)

        elif self.train_agent_id == 2:
            # Training only agent 2
            # Get agent 1's action from its BC model
            p2_action = action
            # Use BC model to get action for agent 1
            p1_observation = self._get_current_observation("agent1")
            p1_action = self._get_bc_action(1, p1_observation)

        else:
            # Training both agents
            p1_action = action["agent1"]
            p2_action = action["agent2"]

        # Execute the step in the game
        observations, rewards, done, info = self.game.step_multi_agent(
            p1_action, p2_action
        )

        # Format return values based on training mode
        if self.train_agent_id == 1:
            return observations["agent1"], rewards["agent1"], done, False, info
        elif self.train_agent_id == 2:
            return observations["agent2"], rewards["agent2"], done, False, info
        else:
            return observations, rewards, done, False, info

    def _get_current_observation(self, agent_key):
        """Helper method to get current observation for a specific agent"""
        player = self.game.player1 if agent_key == "agent1" else self.game.player2
        return self.game._get_agent_observation(for_player=player)

    def _get_bc_action(self, agent_id, observation):
        """
        Get action from pretrained BC model for an agent

        Parameters:
        agent_id: 1 for player 1, 2 for player 2
        observation: The agent's current observation

        Returns:
        Integer action (0-3)
        """
        model = self.p1_model if agent_id == 1 else self.p2_model

        # Extract the part of the observation that the BC model expects
        bc_input_size = model.fc[0].in_features
        bc_input = observation[:bc_input_size]

        # Use the BC model to predict an action
        state_tensor = torch.FloatTensor(bc_input).unsqueeze(0)
        with torch.no_grad():
            logits = model(state_tensor)
            action = torch.argmax(logits, dim=1).item()

        return action

    def get_bc_probs(self, observations):
        """
        Get action probabilities from BC models
        Useful for BC regularization during training

        Parameters:
        observations: Current observations

        Returns:
        Dictionary of BC action probabilities
        """
        return self.game.get_bc_actions(observations)

    def close(self):
        """Clean up resources"""
        self.game.close()
