from collections import deque
import torch
import random
import pygame
from src.player import Player
from src.models import BehaviorCloningModel
from src.tile_definitions import TILE_MAPPING


class RuleBasedAgent(Player):
    def __init__(self, x, y, tile_world, player_id):
        super().__init__(x, y, tile_world, player_id)
        # Set cooldown for AI movement
        self.cooldown = 100  # 500ms
        self.last_step_time = pygame.time.get_ticks()

    def decide_action(self):
        """
        Random decision-making for Rule-Based Agent.
        The agent moves in a random direction if the tile is walkable.
        """
        directions = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
        valid_moves = []
        for d, (dx, dy) in directions.items():
            new_x = self.x + dx
            new_y = self.y + dy
            tile = self.tile_world.get_tile(new_x, new_y)
            if tile:
                # If tile is walkable, add it to valid moves
                if tile.walkable:
                    valid_moves.append((d, dx, dy, tile))
                # If socket is unlocked, allow moving onto it
                elif tile.tile_type == "SOCKET" and self.tile_world.socket_unlocked:
                    valid_moves.append((d, dx, dy, tile))
        if valid_moves:
            return random.choice(valid_moves)
        return None

    def step(self):
        """
        AI Agent's movement logic.
        The agent moves every cooldown period.
        """
        current_time = pygame.time.get_ticks()
        if current_time - self.last_step_time < self.cooldown:
            return
        self.last_step_time = current_time

        action = self.decide_action()
        if action:
            d, dx, dy, tile = action
            self.direction = d
            self.process_move(dx, dy)

    def move(self, event):
        """
        Prevetment of player-controlled movement.
        """
        pass


class TreeBasedAgent(Player):
    def __init__(self, x, y, tile_world, player_id):
        super().__init__(x, y, tile_world, player_id)
        self.cooldown = 500  # Movement cooldown (500ms)
        self.last_step_time = pygame.time.get_ticks()
        self.prev_position = None  # Store last position to avoid backtracking
        self.visited_tiles = {}  # Track visit frequency
        self.recent_moves = deque(maxlen=6)  # Prevent looping
        self.hint_pause = 0  # Pause on hint tiles
        self.target_chip = None  # Coordinates of the next chip to collect

    def find_nearest_chip(self):
        """
        Scan surrounding tiles (up to 4-steps away) to locate the nearest chip.
        If a chip is found, set it as the target.
        If no chip is found, move toward an unexplored tile.
        """
        directions = [
            (0, -1),  # UP
            (0, 1),  # DOWN
            (-1, 0),  # LEFT
            (1, 0),  # RIGHT
        ]

        queue = deque([(self.x, self.y, 0)])  # (x, y, steps taken)
        visited = set()

        while queue:
            x, y, steps = queue.popleft()
            if steps > 4:  # Only scan within 4 steps
                break

            tile = self.tile_world.get_tile(x, y)
            if tile and tile.effect == "COLLECT":
                self.target_chip = (x, y)  # Found a chip!
                return self.target_chip

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and self.tile_world.get_tile(nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny, steps + 1))

        # If no chip found, move towards the least visited tile
        self.target_chip = None
        return None

    def decide_action(self):
        """
        AI decision-making based on:
        1. Finding the nearest chip
        2. Moving toward an unvisited tile if no chip is found
        3. Avoiding hazards & loops
        """
        if self.exited:
            return None  # Do nothing if already exited

        # Check if there is a chip to collect
        if self.target_chip is None or self.target_chip == (self.x, self.y):
            self.find_nearest_chip()

        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        best_move = None
        highest_priority = float("inf")
        possible_moves = []

        for d, (dx, dy) in directions.items():
            new_x, new_y = self.x + dx, self.y + dy
            tile = self.tile_world.get_tile(new_x, new_y)

            if not tile:
                continue  # Skip invalid tiles

            # Allow stepping onto socket if it is unlocked!
            if not tile.walkable and not (
                tile.tile_type == "SOCKET" and self.tile_world.socket_unlocked
            ):
                continue  # Skip non-walkable tiles EXCEPT unlocked socket

            priority = 5  # Default priority for walkable tiles

            # Apply decision tree logic
            if tile.effect == "EXIT" and self.tile_world.socket_unlocked:
                priority = 1  # Exit immediately
            elif tile.effect == "COLLECT":
                priority = 2  # Highest priority (chip collection)
            elif tile.effect == "SOCKET":
                priority = 3 if self.tile_world.socket_unlocked else highest_priority
            elif tile.effect == "HINT":
                priority = 4  # Pause briefly for hint
            elif tile.effect in {"BURN", "DROWN"}:
                priority = highest_priority  # Avoid hazards
            elif tile.effect == "SLIDE":
                priority = 6  # Ice is tricky

            # Move toward target chip if known
            if self.target_chip and (new_x, new_y) == self.target_chip:
                priority = 2  # Prioritize reaching the chip

            # Reduce priority for frequently visited tiles (prevents loops)
            visit_count = self.visited_tiles.get((new_x, new_y), 0)
            priority += visit_count * 2  # Increase penalty for frequent visits

            possible_moves.append((priority, d, dx, dy, tile))

        random.shuffle(possible_moves)  # Randomly shuffle moves first
        possible_moves.sort(key=lambda x: x[0])  # Then sort by priority

        # Pick the best valid move
        for priority, d, dx, dy, tile in possible_moves:
            if priority < float("inf"):
                best_move = (d, dx, dy, tile)
                break

        return best_move

    def step(self):
        """
        AI moves every cooldown period.
        If no chip is visible, agent moves toward unvisited areas.
        """
        current_time = pygame.time.get_ticks()
        if current_time - self.last_step_time < self.cooldown:
            return  # Wait until cooldown period ends

        if self.hint_pause > 0:
            self.hint_pause -= self.cooldown
            return

        self.last_step_time = current_time
        action = self.decide_action()

        if action:
            d, dx, dy, tile = action
            self.direction = d

            # Pause on hint tiles before moving
            if tile.effect == "HINT":
                self.hint_pause = 1500  # 1.5-second pause

            # Store previous position
            self.prev_position = (self.x, self.y)

            # Track visited tiles
            self.visited_tiles[(self.x, self.y)] = (
                self.visited_tiles.get((self.x, self.y), 0) + 1
            )

            # Detect looping & escape cycles
            self.recent_moves.append((self.x, self.y))
            if len(set(self.recent_moves)) <= 3:
                print(f"Loop detected! Forcing random move for Agent {self.player_id}")
                random.shuffle(self.recent_moves)

            # Move the agent
            self.process_move(dx, dy)

    def move(self, event):
        """AI does not use player-controlled movement."""
        pass


class BehaviorClonedAgent(Player):
    def __init__(self, x, y, tile_world, game, player_id, model_path):
        super().__init__(x, y, tile_world, game, player_id)

        # Import the pre-trained Behavior Cloning model
        self.model = self.load_model(model_path)
        self.model.eval()

        # Create reverse mapping (tile_type -> ID) from TILE_MAPPING
        self.tile_type_to_id = {}
        for tile_id, (tile_type, _, _, _, _) in TILE_MAPPING.items():
            self.tile_type_to_id[tile_type] = tile_id

        self.action_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.cooldown = 500  # 500ms
        self.last_step_time = pygame.time.get_ticks()

    def load_model(self, model_path):
        """Load a pre-trained Behavior Cloning model."""
        input_size = 184

        model = BehaviorCloningModel(input_size, 4)
        model.load_state_dict(torch.load(model_path))
        return model

    def get_state_vector(self):
        """Convert player state to tensor format for the ML model."""
        # Get comprehensive state from parent class
        state = super().get_state()

        # Process the full grid
        full_grid = []
        for row in state["full_grid"]:
            processed_row = []
            for tile_type in row:
                # Map to integer using the updated tile definitions
                processed_row.append(self.tile_type_to_id.get(tile_type, 1))
            full_grid.append(processed_row)
        full_grid_tensor = torch.tensor(full_grid, dtype=torch.float32)

        # Convert all components to tensors in the exact order expected by the model
        position = torch.tensor(state["position"], dtype=torch.float32)
        chips_collected = torch.tensor(
            [state["player_collected_chips"]], dtype=torch.float32
        )
        total_chips_collected = torch.tensor(
            [state["total_collected_chips"]], dtype=torch.float32
        )
        socket_unlocked = torch.tensor(
            [int(state["socket_unlocked"])], dtype=torch.float32
        )
        nearest_chip = torch.tensor(state["nearest_chip"], dtype=torch.float32)
        exit_position = torch.tensor(state["exit_position"], dtype=torch.float32)
        is_sliding = torch.tensor([float(state["is_sliding"])], dtype=torch.float32)
        is_being_forced = torch.tensor(
            [float(state["is_being_forced"])], dtype=torch.float32
        )
        alive = torch.tensor([float(state["alive"])], dtype=torch.float32)
        remaining_chips = torch.tensor([state["remaining_chips"]], dtype=torch.float32)
        other_player_pos = torch.tensor(
            state.get("other_player_position", [-1, -1]), dtype=torch.float32
        )

        # Concatenate in the correct order
        state_vector = torch.cat(
            [
                position,
                chips_collected,
                total_chips_collected,
                socket_unlocked,
                nearest_chip,
                exit_position,
                full_grid_tensor.flatten(),
                is_sliding,
                is_being_forced,
                alive,
                remaining_chips,
                other_player_pos,
            ]
        )

        # Add batch dimension
        return state_vector.unsqueeze(0)

    def predict_action(self):
        """Prediction using BC model"""
        with torch.no_grad():
            state_vector = self.get_state_vector()
            output = self.model(state_vector)
            action_idx = torch.argmax(output).item()
        return self.action_mapping[action_idx]

    def step(self):
        """Choose and execute an action after cooldown period"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_step_time < self.cooldown:
            return

        self.last_step_time = current_time

        # Predict the next action
        action = self.predict_action()
        movement = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        if action in movement:
            dx, dy = movement[action]
            self.direction = action
            self.process_move(dx, dy)
