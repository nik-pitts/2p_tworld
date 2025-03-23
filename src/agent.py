from collections import deque
import heapq
import torch
import random
import pygame
from src.player import Player
from src.models import BehaviorCloningModel, BehaviorCloningModelLv2
from src.tile_definitions import TILE_MAPPING
import torch.nn.functional as F


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
    def __init__(
        self,
        x,
        y,
        tile_world,
        game,
        player_id,
        model_path,
        is_train=False,
        alignment=0,
    ):
        super().__init__(x, y, tile_world, game, player_id)

        # Import the pre-trained Behavior Cloning model
        self.model = self.load_model(model_path)
        self.model.eval()

        # Create reverse mapping (tile_type -> ID) from TILE_MAPPING
        self.tile_type_to_id = {}
        for tile_id, (tile_type, _, _, _, _) in TILE_MAPPING.items():
            self.tile_type_to_id[tile_type] = tile_id

        self.action_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.cooldown = 400  # 500ms
        self.last_step_time = pygame.time.get_ticks()
        self.is_train = is_train
        self.alignment = alignment  # 0: aligned, 1: merged, 2: diverged
        self.set_child_attribute(self.is_train)  # Pass child's attribute to parent

        # Add only essential A* pathfinding variables
        self.path = []  # Queue of moves to follow
        self.current_target = None

    def load_model(self, model_path):
        """Load a pre-trained Behavior Cloning model."""
        input_size = 191

        model = BehaviorCloningModel(input_size, 4)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

    def get_state_vector(self):
        """Convert player state to tensor format for the ML model."""
        # Get comprehensive state from parent class
        state = super().get_state()

        # Process the fusll grid
        full_grid = []
        for row in state["full_grid"]:
            processed_row = []
            for tile_type in row:
                # Map to integer using the updated tile definitions
                processed_row.append(self.tile_type_to_id.get(tile_type, 1))
            full_grid.append(processed_row)
        full_grid_tensor = torch.tensor(full_grid, dtype=torch.float32)

        # Process the local grid
        local_grid = []
        for row in state["local_grid"]:
            processed_row = []
            for tile_type in row:
                # Map to integer using the updated tile definitions
                processed_row.append(self.tile_type_to_id.get(tile_type, 1))
            local_grid.append(processed_row)
        local_grid_tensor = torch.tensor(local_grid, dtype=torch.float32)

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
        exit_location = torch.tensor(state["exit_position"], dtype=torch.float32)
        alive = torch.tensor([float(state["alive"])], dtype=torch.float32)
        remaining_chips = torch.tensor([state["remaining_chips"]], dtype=torch.float32)
        other_player_pos = torch.tensor(
            state.get("other_player_position", [-1, -1]), dtype=torch.float32
        )

        state_vector = torch.cat(
            [
                position,
                chips_collected,
                total_chips_collected,
                socket_unlocked,
                nearest_chip,
                exit_location,
                full_grid_tensor.flatten(),
                local_grid_tensor.flatten(),
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

    def step(self, action=None):
        movement = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        if not self.is_train:
            """Choose and execute an action after cooldown period"""
            current_time = pygame.time.get_ticks()
            if current_time - self.last_step_time < self.cooldown:
                return
            self.last_step_time = current_time

            # Check if we need to update our path
            if not self.path:
                # Get my assignments
                my_assignments = self.get_my_assignments()
                # If we have assignments, use A* pathfinding
                if my_assignments:
                    print(f"Agent {self.player_id} has assignments: {my_assignments}")
                    # Sort assignments by distance to find closest target
                    my_assignments.sort(
                        key=lambda pos: self.manhattan_distance((self.x, self.y), pos)
                    )
                    self.current_target = my_assignments[0]

                    # Calculate path to target
                    self.path = self.a_star_search(
                        (self.x, self.y), self.current_target
                    )

            # If we have a path to follow (A* mode)
            if self.path:
                # Get next direction from path
                next_move = self.path[0]
                self.path = self.path[1:]  # Remove this step

                # Execute the move
                self.direction = next_move
                dx, dy = movement[next_move]
                self.process_move(dx, dy)

                # If we reached the target, clear it
                if self.current_target and (self.x, self.y) == self.current_target:
                    self.current_target = None
                    self.path = []

                return

        # Predict the next action
        if action is None:
            action = self.predict_action()

        if action in movement:
            dx, dy = movement[action]
            self.direction = action
            self.process_move(dx, dy)
        else:
            pass

    def manhattan_distance(self, a, b):
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_search(self, start, goal):
        """
        A* pathfinding algorithm to find the optimal path from start to goal.

        Args:
            start (tuple): (x, y) starting position
            goal (tuple): (x, y) goal position

        Returns:
            list: List of directions to take from start to goal
        """
        # Check if goal is valid
        goal_tile = self.tile_world.get_tile(goal[0], goal[1])
        if not goal_tile:
            return []

        # Define directions: Up, Down, Left, Right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        direction_names = ["UP", "DOWN", "LEFT", "RIGHT"]

        # Initialize open and closed sets
        open_set = []
        heapq.heappush(open_set, (0, start, []))  # (f_score, position, path)
        closed_set = set()

        # g_score: cost from start to current
        g_score = {start: 0}

        # Limit search to prevent infinite loops
        max_iterations = 1000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            # Get node with lowest f_score
            f, current, path = heapq.heappop(open_set)

            # If we reached the goal, return the path
            if current == goal:
                return path

            # Skip if already processed
            if current in closed_set:
                continue

            # Mark as processed
            closed_set.add(current)

            # Try each direction
            for i, (dx, dy) in enumerate(directions):
                neighbor = (current[0] + dx, current[1] + dy)

                # Check if the neighbor is valid
                if not (
                    0 <= neighbor[0] < self.tile_world.width
                    and 0 <= neighbor[1] < self.tile_world.height
                ):
                    continue

                # Check if the neighbor is walkable
                neighbor_tile = self.tile_world.get_tile(neighbor[0], neighbor[1])
                if not neighbor_tile or not neighbor_tile.walkable:
                    # Special case: allow socket if unlocked
                    if (
                        neighbor_tile
                        and neighbor_tile.tile_type == "SOCKET"
                        and self.tile_world.socket_unlocked
                    ):
                        pass  # Allow this move
                    else:
                        continue  # Invalid or unwalkable

                # Handle hazards (fire and water)
                if neighbor_tile.effect == "BURN" and not self.boots.get("FIRE", False):
                    continue  # Skip fire tiles if we don't have fire boots

                if neighbor_tile.effect == "DROWN" and not self.boots.get(
                    "WATER", False
                ):
                    continue  # Skip water tiles if we don't have water boots

                # Calculate tentative g_score
                tentative_g = g_score[current] + 1

                # If this path is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.manhattan_distance(neighbor, goal)

                    # Add to open set with updated path
                    new_path = path + [direction_names[i]]
                    heapq.heappush(open_set, (f_score, neighbor, new_path))

        # No path found
        return []

    def get_my_assignments(self):
        """Get assignments for this player only from the UI."""
        # Check if UI has item assignments
        if hasattr(self.game, "ui") and hasattr(self.game.ui, "item_assignments"):
            item_assignments = self.game.ui.item_assignments
            my_assignments = []

            # Format: {(x, y): (player_id, sprite)}
            for pos, assignment_info in item_assignments.items():
                player_id = assignment_info[0]  # First element is player_id
                if player_id == self.player_id:
                    my_assignments.append(pos)

            return my_assignments

        # No assignments found
        return []


class BehaviorClonedAgentLv2(Player):
    def __init__(
        self,
        x,
        y,
        tile_world,
        game,
        player_id,
        model_path,
        is_train=False,
        alignment=0,
        normalize_features=False,
    ):
        super().__init__(x, y, tile_world, game, player_id)

        # Import the pre-trained Behavior Cloning model
        self.model = self.load_model(model_path)
        self.model.eval()

        # Create reverse mapping (tile_type -> ID) from TILE_MAPPING
        self.tile_type_to_id = {}
        for tile_id, (tile_type, _, _, effect, _) in TILE_MAPPING.items():
            # Handle special cases with effects
            if effect and effect.startswith("BEETLE_"):
                full_type = f"BEETLE_{effect.split('_')[1]}"
                self.tile_type_to_id[full_type] = tile_id
            elif effect and effect.startswith("FORCE_"):
                self.tile_type_to_id["FORCE_FLOOR"] = tile_id
            else:
                self.tile_type_to_id[tile_type] = tile_id

        # Action mapping including special states
        self.action_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        self.cooldown = 250  # 500ms between actions
        self.last_step_time = pygame.time.get_ticks()
        self.is_train = is_train
        self.alignment = alignment  # 0: aligned, 1: merged, 2: diverged

        # Feature normalization option
        self.normalize_features = normalize_features

        # Calculate dataset statistics if normalizing
        if self.normalize_features:
            self.calculate_normalization_stats()

        # Debug mode for development
        self.debug_mode = False

        # Map for key and boot types
        self.key_mapping = {"RED": 0, "BLUE": 1, "YELLOW": 2, "GREEN": 3}
        self.boot_mapping = {"WATER": 0, "FIRE": 1, "FORCE": 2}

    def load_model(self, model_path):
        # First examine the model architecture
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

        # Initialize model with the correct architecture
        model = BehaviorCloningModelLv2(209, 4)

        # Load the weights
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
        return model

    def calculate_normalization_stats(self):
        """Calculate statistics for feature normalization"""
        # Extract grid dimensions from tile_world
        self.grid_height = self.tile_world.height
        self.grid_width = self.tile_world.width

        # Initialize normalization ranges
        self.position_max = max(self.grid_height, self.grid_width)
        self.chip_max = (
            self.tile_world.total_chips
            if hasattr(self.tile_world, "total_chips")
            else 2
        )

    def get_state_vector(self):
        """Create a comprehensive state vector for the model"""
        # Get comprehensive state from parent class
        state = super().get_state()

        # Extract state components
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
        nearest_key = torch.tensor(state["nearest_key"], dtype=torch.float32)
        nearest_boot = torch.tensor(state["nearest_boot"], dtype=torch.float32)
        exit_location = torch.tensor(state["exit_position"], dtype=torch.float32)

        # Process key and boot information
        # Convert dictionary form to binary features
        # Keys and boots (encode as one-hot vectors)
        keys_tensor = torch.zeros(4, dtype=torch.float32)  # RED, BLUE, YELLOW, GREEN
        if state["collected_keys"]:
            for key in state["collected_keys"]:
                if key in self.key_mapping:
                    keys_tensor[self.key_mapping[key]] = 1

        boots_tensor = torch.zeros(3, dtype=torch.float32)  # WATER, FIRE, FORCE
        if state["collected_boots"]:
            for boot in state["collected_boots"]:
                if boot in self.boot_mapping:
                    boots_tensor[self.boot_mapping[boot]] = 1

        # Process the full grid - convert tile types to their IDs
        full_grid = []
        for row in state["full_grid"]:
            processed_row = []
            for tile_type in row:
                # Map to integer using the tile_type_to_id mapping
                tile_id = self.tile_type_to_id.get(
                    tile_type, self.tile_type_to_id.get("WALL", 1)
                )
                processed_row.append(tile_id)
            full_grid.append(processed_row)

        full_grid_tensor = torch.tensor(full_grid, dtype=torch.float32)

        # Process the local grid - convert tile types to their IDs
        local_grid = []
        for row in state["local_grid"]:
            grid_row = []
            for tile_type in row:
                # Default to WALL if type not in mapping
                tile_value = self.tile_type_to_id.get(tile_type, 1)
                grid_row.append(tile_value)
            local_grid.append(grid_row)
        local_grid_tensor = torch.tensor(local_grid, dtype=torch.float32).flatten()

        # Augmented local grid. To large
        # local_grid_tensor = torch.tensor(state["local_grid"], dtype=torch.float32)

        # Normalize grid values if enabled
        if self.normalize_features:
            max_tile_id = max(self.tile_type_to_id.values())
            full_grid_tensor = full_grid_tensor / max_tile_id
            local_grid_tensor = local_grid_tensor / max_tile_id

        # Additional state information
        is_sliding = torch.tensor(
            [float(state.get("is_sliding", False))], dtype=torch.float32
        )
        is_being_forced = torch.tensor(
            [float(state.get("is_being_forced", False))], dtype=torch.float32
        )
        alive = torch.tensor([float(state.get("alive", True))], dtype=torch.float32)
        remaining_chips = torch.tensor(
            [state.get("remaining_chips", 0)], dtype=torch.float32
        )
        other_player_pos = torch.tensor(
            state.get("other_player_position", [-1, -1]), dtype=torch.float32
        )
        player_id = torch.tensor([state["player_id"]], dtype=torch.float32)

        # Time information
        time_elapsed = torch.tensor(
            [state["time_elapsed"] / 1000] if "time_elapsed" in state else [0],
            dtype=torch.float32,
        )  # Normalize to seconds

        # Goal position information
        goal_pos = torch.tensor(
            state["goal_pos"] if "goal_pos" in state else [-1, -1], dtype=torch.float32
        )

        # Other player information
        other_player_chips = torch.tensor(
            [state["other_player_collected_chips"]]
            if "other_player_collected_chips" in state
            else [0],
            dtype=torch.float32,
        )

        # Normalize position-based features if enabled
        if self.normalize_features:
            position = position / self.position_max
            nearest_chip = nearest_chip / self.position_max
            exit_location = exit_location / self.position_max

            if not (other_player_pos[0] == -1 and other_player_pos[1] == -1):
                other_player_pos = other_player_pos / self.position_max

            chips_collected = chips_collected / self.chip_max
            total_chips_collected = total_chips_collected / self.chip_max
            remaining_chips = remaining_chips / self.chip_max

        # Concatenate all state information into a single vector
        state_vector = torch.cat(
            [
                position,  # 2 values
                chips_collected,  # 1 value
                total_chips_collected,  # 1 value
                socket_unlocked,  # 1 value
                nearest_chip,  # 2 values
                nearest_key,
                nearest_boot,
                exit_location,  # 2 values
                full_grid_tensor.flatten(),  # grid_height * grid_width values
                local_grid_tensor.flatten(),  # 5*5 = 25 values for local grid
                is_sliding,  # 1 value
                is_being_forced,  # 1 value
                alive,  # 1 value
                remaining_chips,  # 1 value
                other_player_pos,  # 2 values
                player_id,
                keys_tensor,
                boots_tensor,
                time_elapsed,
                goal_pos,
                other_player_chips,
            ]
        )

        # Debug information if needed
        if self.debug_mode:
            print(f"State vector shape: {state_vector.shape}")

        # Add batch dimension for model input
        return state_vector.unsqueeze(0)

    def predict_action(self):
        """Predict the next action using the behavior cloning model"""

        with torch.no_grad():
            # Get the state vector
            state_vector = self.get_state_vector()

            # Forward pass through the model
            output = self.model(state_vector)

            # Get the action with highest probability
            action_idx = torch.argmax(output, dim=1).item()

            # Debug information if needed
            if self.debug_mode:
                probs = F.softmax(output, dim=1)
                print(f"Action probabilities: {probs}")
                print(f"Selected action: {self.action_mapping[action_idx]}")

            return self.action_mapping[action_idx]

    def step(self):
        """Execute an action after cooldown period"""
        # Check if it's time to take another action
        current_time = pygame.time.get_ticks()
        if current_time - self.last_step_time < self.cooldown:
            return  # Still in cooldown period

        # Update the cooldown timestamp
        self.last_step_time = current_time

        # If agent is being forced or sliding, don't make a decision
        if self.is_being_forced or self.is_sliding:
            return

        # Predict the next action
        action = self.predict_action()

        # print(action)

        # Movement mappings
        valid_movement = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        # Execute the action if it's a directional movement
        if action in valid_movement:
            dx, dy = valid_movement[action]
            self.direction = action
            self.process_move(dx, dy)


class RLAgent(Player):
    def __init__(
        self,
        x,
        y,
        tile_world,
        game,
        player_id,
        model_path=None,
        is_train=False,
        alignment=None,
    ):
        super().__init__(x, y, tile_world, game, player_id)

        # For trained models that will be deployed (not during training)
        self.sb3_model = None
        if model_path and not is_train:
            from stable_baselines3 import PPO

            self.sb3_model = PPO.load(model_path)

        # Create reverse mapping (tile_type -> ID) from TILE_MAPPING
        self.tile_type_to_id = {}
        for tile_id, (tile_type, _, _, _, _) in TILE_MAPPING.items():
            self.tile_type_to_id[tile_type] = tile_id

        self.action_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.reverse_action_mapping = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        self.cooldown = 500  # 500ms
        self.last_step_time = pygame.time.get_ticks()
        self.is_train = is_train
        self.set_child_attribute(self.is_train)  # Pass child's attribute to parent

        # Initialize tracking variables for reward calculation
        self.prev_collected_chips = 0
        self.prev_position = (self.x, self.y)
        self.stuck_count = 0

    def get_state_vector(self):
        """Convert player state to tensor format (same format as BehaviorClonedAgent)"""
        # Get comprehensive state from parent class
        state = super().get_state()

        # Process the full grid
        # full_grid = []
        # for row in state["full_grid"]:
        #     processed_row = []
        #     for tile_type in row:
        #         # Map to integer using the updated tile definitions
        #         processed_row.append(self.tile_type_to_id.get(tile_type, 1))
        #     full_grid.append(processed_row)
        # full_grid_tensor = torch.tensor(full_grid, dtype=torch.float32)

        # Process the local grid
        local_grid = []
        for row in state["local_grid"]:
            processed_row = []
            for tile_type in row:
                # Map to integer using the updated tile definitions
                processed_row.append(self.tile_type_to_id.get(tile_type, 1))
            local_grid.append(processed_row)
        local_grid_tensor = torch.tensor(local_grid, dtype=torch.float32)

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
        chip_positions = torch.tensor(state["chip_positions"], dtype=torch.float32)
        exit_location = torch.tensor(state["exit_position"], dtype=torch.float32)
        alive = torch.tensor([float(state["alive"])], dtype=torch.float32)
        remaining_chips = torch.tensor([state["remaining_chips"]], dtype=torch.float32)
        other_player_pos = torch.tensor(
            state.get("other_player_position", [-1, -1]), dtype=torch.float32
        )

        # Concatenate all features
        state_vector = torch.cat(
            [
                position,
                chips_collected,
                total_chips_collected,
                socket_unlocked,
                nearest_chip,
                chip_positions.flatten(),
                exit_location,
                # full_grid_tensor.flatten(),
                local_grid_tensor.flatten(),
                alive,
                remaining_chips,
                other_player_pos,
            ]
        )

        return state_vector

    def predict_action(self):
        """Use trained SB3 model to predict action (used during deployment)"""
        if self.sb3_model:
            with torch.no_grad():
                # Get state vector without batch dimension
                state_vector = self.get_state_vector().numpy()
                # Use SB3 model to predict
                action, _ = self.sb3_model.predict(state_vector, deterministic=True)
                return self.action_mapping[action.item()]

    def step(self, action=None):
        """Execute step with either provided action or model-predicted action"""
        if not self.is_train:
            # During deployment, check cooldown
            current_time = pygame.time.get_ticks()
            if current_time - self.last_step_time < self.cooldown:
                return
            self.last_step_time = current_time

            # If no explicit action provided, use model prediction
            if action is None:
                action = self.predict_action()

        # Map action string to movement
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

        # Update tracking variables for reward calculation
        self.prev_collected_chips = self.collected_chips
        new_position = (self.x, self.y)
        if new_position == self.prev_position:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.prev_position = new_position
