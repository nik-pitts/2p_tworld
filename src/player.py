import pygame
import time
from collections import deque
from src.tiles import Tile
import src.settings as settings


class Player:
    def __init__(self, x, y, tile_world, game, player_id, record=False):
        self.x, self.y = x, y
        self.tile_size = settings.TILE_SIZE
        self.direction = "DOWN"
        self.speed = 1
        self.tile_world = tile_world
        self.game = game
        self.player_id = player_id
        self.exited = False
        self.collected_chips = 0
        self.human_play_data = []
        self.record = record
        self.alive = True
        self.slide_dx, self.slide_dy = 0, 0
        self.is_train = False
        self.clicked_optime = False

        # force floor handling variables
        self.is_being_forced = False
        self.force_movement_queue = []
        self.force_move_time = 0
        self.force_move_delay = 80  # milliseconds between steps

        # ice floor handling variables
        self.is_sliding = False
        self.slide_movement_queue = []
        self.slide_move_time = 0
        self.slide_move_delay = 100  # milliseconds between steps

        # Key tracking
        self.keys = {"RED": False, "BLUE": False, "YELLOW": False, "GREEN": False}

        # Boot tracking
        self.boots = {"WATER": False, "FIRE": False, "FORCE": False}

        # Key mappings for player controls
        self.controls = {
            1: {
                "UP": pygame.K_w,
                "DOWN": pygame.K_s,
                "LEFT": pygame.K_a,
                "RIGHT": pygame.K_d,
                "STILL": pygame.K_SPACE,
            },
            2: {
                "UP": pygame.K_UP,
                "DOWN": pygame.K_DOWN,
                "LEFT": pygame.K_LEFT,
                "RIGHT": pygame.K_RIGHT,
                "STILL": pygame.K_SPACE,
            },
        }[player_id]

        # Load tile sprite sheet
        self.tile_sprite_sheet = tile_world.sprite_sheet

        # Player sprites
        base_row = 12
        self.sprites = {
            "UP": self.tile_sprite_sheet.get_tile(6, base_row),
            "LEFT": self.tile_sprite_sheet.get_tile(6, base_row + 1),
            "DOWN": self.tile_sprite_sheet.get_tile(6, base_row + 2),
            "RIGHT": self.tile_sprite_sheet.get_tile(6, base_row + 3),
        }

        self.image = self.sprites["DOWN"]

    def process_move(self, dx, dy):
        """Process movement and log all outcomes.

        Args:
            dx: Change in x position
            dy: Change in y position
        """
        # Capture initial state before move
        initial_state = self.get_state() if self.record else None
        move_outcome = "attempted"
        # initial_position = (self.x, self.y)

        # Early exit conditions
        if not self.alive or self.exited:
            move_outcome = "blocked_dead_or_exited"
            if self.record:
                self.log_move(self.direction, initial_state, move_outcome)
            return

        if self.is_sliding:
            move_outcome = "blocked_sliding"
            if self.record:
                self.log_move(self.direction, initial_state, move_outcome)
            return

        # Calculate new position
        new_x, new_y = self.x + dx, self.y + dy
        tile = self.tile_world.get_tile(new_x, new_y)

        # if not self.is_train:
        # Remove hint when moving
        # self.tile_world.game_ui.clear_hint()

        # Socket tile handling
        if tile.tile_type == "SOCKET":
            if not self.tile_world.socket_unlocked:
                # print(f"Player {self.player_id} cannot pass the socket yet!")
                move_outcome = "blocked_socket_locked"
                if self.record:
                    self.log_move(self.direction, initial_state, move_outcome)
                return
            else:
                # print(f"Player {self.player_id} stepped on the socket! Removing it.")
                self.tile_world.remove_socket(new_x, new_y)
                self.x, self.y = new_x, new_y
                move_outcome = "success_socket_removed"
                if self.record:
                    self.log_move(self.direction, initial_state, move_outcome)
                return

        # Death conditions - water and fire
        if tile.effect in ["BURN", "DROWN"]:
            # Check if player has corresponding boots
            has_protection = (tile.effect == "BURN" and self.boots["FIRE"]) or (
                tile.effect == "DROWN" and self.boots["WATER"]
            )

            if has_protection:
                # Player has appropriate boots, move normally
                # print(
                #     f"Player {self.player_id}'s boots protected them from {tile.effect.lower()}ing!"
                # )
                self.x, self.y = new_x, new_y
                move_outcome = f"success_protected_from_{tile.effect.lower()}"
            else:
                # Player doesn't have boots, they die
                print(f"Player {self.player_id} {tile.effect.lower()}ed!")
                self.remove_self()
                self.game.check_game_over()
                move_outcome = f"death_{tile.effect.lower()}"

            if self.record:
                self.log_move(self.direction, initial_state, move_outcome)
            return

        # Ice sliding mechanic
        if tile.effect == "SLIDE":
            move_outcome = "slide_initiated"
            if self.record:
                self.log_move(self.direction, initial_state, move_outcome)
            self.slide_on_ice(dx, dy)
            return

        # Force floor mechanic
        if tile.effect and tile.effect.startswith("FORCE_"):
            if self.boots["FORCE"]:
                # Player has force boots, just move normally
                print(
                    f"Player {self.player_id}'s force boots prevented them from being moved!"
                )
                self.x, self.y = new_x, new_y
                move_outcome = "success_force_resisted"
            else:
                # Player doesn't have force boots, they get moved
                print(f"Player {self.player_id} stepped on a force floor!")
                self.x, self.y = new_x, new_y
                self.force_by_floor(tile.effect)
                move_outcome = f"forced_{tile.effect}"

            if self.record:
                self.log_move(self.direction, initial_state, move_outcome)
            return

        # Hint tile handling
        if tile.effect == "HINT":
            # print(f"Player {self.player_id} found a hint: {self.tile_world.hint}")
            self.tile_world.game_ui.show_hint(self.tile_world.hint)
            move_outcome = "success_hint"

        # Chip collection handling
        if tile.effect == "COLLECT":
            self.collected_chips += 1
            self.tile_world.collect_chip()
            self.tile_world.set_tile(
                new_x,
                new_y,
                Tile(
                    "FLOOR",
                    walkable=True,
                    effect=None,
                    sprite_sheet=self.tile_world.sprite_sheet,
                    sprite_index=(0, 0),
                ),
            )

            self.tile_world.remove_collectable(new_x, new_y)
            collected_position = (new_x, new_y)
            self.tile_world.game_ui.update_inventory(self, tile, collected_position)
            move_outcome = "success_collect_chip"

        # Key collection
        if tile.tile_type.startswith("KEY"):
            key_color = tile.effect  # This should be "RED", "BLUE", etc.
            self.tile_world.collect_key(key_color, self)
            self.tile_world.set_tile(
                new_x,
                new_y,
                Tile(
                    "FLOOR",
                    walkable=True,
                    effect=None,
                    sprite_sheet=self.tile_world.sprite_sheet,
                    sprite_index=(0, 0),
                ),
            )

            # Remove key from key_positions list
            if (new_x, new_y) in self.tile_world.key_positions:
                self.tile_world.key_positions.remove((new_x, new_y))
                print(self.tile_world.key_positions)

            # Update UI inventory
            self.tile_world.remove_collectable(new_x, new_y)
            collected_position = (new_x, new_y)
            self.tile_world.game_ui.update_inventory(self, tile, collected_position)
            move_outcome = f"success_collect_key_{key_color}"

        # Boot collection
        if tile.tile_type.startswith("BOOT"):
            boot_type = tile.effect  # This will be "WATER", "FIRE", or "FORCE"
            self.boots[boot_type] = True
            # print(f"👢 Player {self.player_id} collected {boot_type} boots!")
            self.tile_world.set_tile(
                new_x,
                new_y,
                Tile(
                    "FLOOR",
                    walkable=True,
                    effect=None,
                    sprite_sheet=self.tile_world.sprite_sheet,
                    sprite_index=(0, 0),
                ),
            )

            # Remove key from key_positions list
            if (new_x, new_y) in self.tile_world.boot_positions:
                self.tile_world.boot_positions.remove((new_x, new_y))

            # Update UI inventory
            self.tile_world.remove_collectable(new_x, new_y)
            collected_position = (new_x, new_y)
            self.tile_world.game_ui.update_inventory(self, tile, collected_position)
            move_outcome = f"success_collect_boot_{boot_type}"

        # Door handling
        if tile.tile_type.startswith("DOOR"):
            door_color = tile.effect  # This should match the key color required
            if self.keys.get(door_color, False):
                # print(f"Player {self.player_id} unlocked a {door_color} door!")
                self.tile_world.set_tile(
                    new_x,
                    new_y,
                    Tile(
                        "FLOOR",
                        walkable=True,
                        effect=None,
                        sprite_sheet=self.tile_world.sprite_sheet,
                        sprite_index=(0, 0),
                    ),
                )
                # Now the player can move to this position
                self.x, self.y = new_x, new_y
                move_outcome = f"success_unlock_door_{door_color}"

                if self.record:
                    self.log_move(self.direction, initial_state, move_outcome)
                return
            else:
                print(
                    f"Player {self.player_id} needs a {door_color} key to unlock this door!"
                )
                move_outcome = f"blocked_door_needs_key_{door_color}"

                if self.record:
                    self.log_move(self.direction, initial_state, move_outcome)
                return

        # Block pushing
        if tile.effect == "PUSH":
            push_result = self.push_block(new_x, new_y, dx, dy)
            if push_result:
                move_outcome = "success_pushed_block"
            else:
                move_outcome = "blocked_could_not_push"

            if self.record:
                self.log_move(self.direction, initial_state, move_outcome)
            return

        # Exit handling
        if tile.effect == "EXIT":
            if self.tile_world.socket_unlocked:
                # print(f"Player {self.player_id} escaped!")
                self.exited = True
                self.alive = False
                move_outcome = "success_exit"
            else:
                move_outcome = "blocked_exit_socket_locked"

            if self.record:
                self.log_move(self.direction, initial_state, move_outcome)
            return

        # Wall collision detection
        if tile.walkable:
            self.x, self.y = new_x, new_y
            move_outcome = "success_move"
        else:
            move_outcome = "blocked_unwalkable"

        # Update player sprite based on direction
        if self.direction != "STILL":
            self.image = self.sprites[self.direction]

        # Log the final outcome if recording is enabled
        if self.record:
            self.log_move(self.direction, initial_state, move_outcome)

    def slide_on_ice(self, dx, dy):
        """Prepares the player to slide on ice continuously until hitting a non-slide tile."""
        # Set up sliding state
        self.is_sliding = True
        self.slide_movement_queue = []

        # Set direction based on initial slide direction
        if dx == 1:
            self.direction = "RIGHT"
        elif dx == -1:
            self.direction = "LEFT"
        elif dy == 1:
            self.direction = "DOWN"
        elif dy == -1:
            self.direction = "UP"

        self.image = self.sprites[self.direction]

        # Calculate all slide positions ahead of time
        current_x, current_y = self.x, self.y

        while True:
            next_x, next_y = current_x + dx, current_y + dy
            next_tile = self.tile_world.get_tile(next_x, next_y)

            # Stop if next tile isn't valid
            if not next_tile or not next_tile.walkable:
                break

            # Handle death tiles
            if next_tile.effect in ["BURN", "DROWN"]:
                self.slide_movement_queue.append((next_x, next_y, next_tile.effect))
                break

            # Add position to queue
            self.slide_movement_queue.append((next_x, next_y, None))
            current_x, current_y = next_x, next_y

            # Check if we should continue sliding
            if next_tile.effect != "SLIDE":
                break

        # Initialize the timer for movement
        self.slide_move_time = pygame.time.get_ticks()
        self.slide_move_delay = 100  # milliseconds between steps

    def update_sliding_movement(self):
        """Update player position during sliding. Returns True if a move was made."""
        if not self.is_sliding or not self.slide_movement_queue:
            return False

        current_time = pygame.time.get_ticks()

        # Check if it's time for the next movement
        if current_time - self.slide_move_time >= self.slide_move_delay:
            self.slide_move_time = current_time

            # Get next position
            next_x, next_y, effect = self.slide_movement_queue.pop(0)

            # Handle death effects
            if effect in ["BURN", "DROWN"]:
                # print(f"Player {self.player_id} {effect.lower()}ed while sliding!")
                self.remove_self()
                if self.record:
                    self.log_move(self.direction, None, f"death_slide_{effect.lower()}")
                self.game.check_game_over()
                self.is_sliding = False
                return True

            # Move player
            self.x, self.y = next_x, next_y

            if self.record:
                self.log_move(self.direction, None, "slide_continue")

            # Check if sliding is complete
            if not self.slide_movement_queue:
                self.is_sliding = False
                if self.record:
                    self.log_move(self.direction, None, "slide_complete")

            return True

        return False

    def force_by_floor(self, initial_force_direction):
        """Moves the player continuously according to force floor direction."""
        # Add this to your player class to track force movement
        self.is_being_forced = True
        self.force_movement_queue = []

        force_map = {
            "FORCE_UP": (0, -1),
            "FORCE_DOWN": (0, 1),
            "FORCE_LEFT": (-1, 0),
            "FORCE_RIGHT": (1, 0),
        }

        direction_map = {
            "FORCE_UP": "UP",
            "FORCE_DOWN": "DOWN",
            "FORCE_LEFT": "LEFT",
            "FORCE_RIGHT": "RIGHT",
        }

        self.direction = direction_map.get(initial_force_direction, self.direction)
        self.image = self.sprites[self.direction]  # Update the player sprite

        force_direction = initial_force_direction
        current_x, current_y = self.x, self.y

        # Calculate all force movement positions first
        while True:
            dx, dy = force_map[force_direction]
            next_x, next_y = current_x + dx, current_y + dy
            next_tile = self.tile_world.get_tile(next_x, next_y)

            # Stop if the next tile is not walkable
            if not next_tile or not next_tile.walkable:
                break

            # Store death tile info if applicable
            if next_tile.effect in ["BURN", "DROWN"]:
                self.force_movement_queue.append((next_x, next_y, next_tile.effect))
                break

            # Add this position to the queue
            self.force_movement_queue.append((next_x, next_y, None))
            current_x, current_y = next_x, next_y

            # Update force direction if on another force floor
            if next_tile.effect and next_tile.effect.startswith("FORCE_"):
                force_direction = next_tile.effect
                self.direction = direction_map.get(force_direction, self.direction)
            else:
                break

        # Start force movement animation clock
        self.force_move_time = pygame.time.get_ticks()
        self.force_move_delay = 80  # milliseconds between steps

        if self.record:
            self.log_move(self.direction, None, f"force_movement_{self.direction}")

    def update_forced_movement(self):
        """Update the player's position during forced movement"""
        if not self.is_being_forced or not self.force_movement_queue:
            return False

        current_time = pygame.time.get_ticks()

        # Check if it's time for the next movement
        if current_time - self.force_move_time >= self.force_move_delay:
            self.force_move_time = current_time

            # Get next position from queue
            next_x, next_y, effect = self.force_movement_queue.pop(0)

            # Handle death effects
            if effect in ["BURN", "DROWN"]:
                # print(f"Player {self.player_id} {effect.lower()}ed while being forced!")
                self.remove_self()

                if self.record:
                    self.log_move(
                        self.direction, None, f"death_forced_{effect.lower()}"
                    )

                self.game.check_game_over()
                self.is_being_forced = False
                return True

            # Detect direction change for new force floors
            tile = self.tile_world.get_tile(next_x, next_y)
            if tile and tile.effect and tile.effect.startswith("FORCE_"):
                direction_map = {
                    "FORCE_UP": "UP",
                    "FORCE_DOWN": "DOWN",
                    "FORCE_LEFT": "LEFT",
                    "FORCE_RIGHT": "RIGHT",
                }
                new_direction = direction_map.get(tile.effect, self.direction)
                if new_direction != self.direction:
                    self.direction = new_direction
                    self.image = self.sprites[self.direction]

                    # Log direction change
                    if self.record:
                        self.log_move(
                            self.direction,
                            None,
                            f"force_direction_change_{self.direction}",
                        )

            # Move player to next position
            self.x, self.y = next_x, next_y

            if self.record:
                self.log_move(self.direction, None, "force_movement")

            # If queue is empty, end forced movement
            if not self.force_movement_queue:
                self.is_being_forced = False

            return True

        return False

    def push_block(self, block_x, block_y, dx, dy):
        """Handles pushing a movable block. Returns True if push was successful."""
        target_x, target_y = block_x + dx, block_y + dy
        next_tile = self.tile_world.get_tile(target_x, target_y)

        if next_tile and next_tile.walkable:
            print(
                f"Player {self.player_id} pushed a block from ({block_x}, {block_y}) to ({target_x}, {target_y})"
            )

            if next_tile.tile_type in ["WATER", "FIRE", "FORCE_FLOOR"]:
                self.tile_world.set_tile(
                    target_x,
                    target_y,
                    Tile("DIRT", True, None, self.tile_world.sprite_sheet, (0, 11)),
                )
            else:
                self.tile_world.set_tile(
                    target_x,
                    target_y,
                    Tile(
                        "MOVABLE_DIRT_BLOCK",
                        False,
                        "PUSH",
                        self.tile_world.sprite_sheet,
                        (0, 10),
                    ),
                )

            # Replace old block position with floor
            self.tile_world.set_tile(
                block_x,
                block_y,
                Tile("FLOOR", True, None, self.tile_world.sprite_sheet, (0, 0)),
            )
            self.x, self.y = block_x, block_y  # Allow player to move forward
            return True
        return False

    def move(self, event):
        if self.exited or not self.alive:
            return

        # Get the current tile the player is standing on
        current_tile = self.tile_world.get_tile(self.x, self.y)

        # If the player is on an ice tile, they can't control movement
        if current_tile and current_tile.effect == "SLIDE":
            return

        dx, dy = 0, 0
        if event.key == self.controls["UP"]:
            dy = -self.speed
            self.direction = "UP"
        elif event.key == self.controls["DOWN"]:
            dy = self.speed
            self.direction = "DOWN"
        elif event.key == self.controls["LEFT"]:
            dx = -self.speed
            self.direction = "LEFT"
        elif event.key == self.controls["RIGHT"]:
            dx = self.speed
            self.direction = "RIGHT"
        elif event.key == self.controls["STILL"]:
            dx, dy = 0, 0
            self.direction = "STILL"

        self.process_move(dx, dy)

    def log_move(self, direction, state=None, outcome="success"):
        """Log a move with its outcome and state information."""
        # Capture complete state if not provided
        if state is None:
            state = self.get_state()

        state["alive"] = self.alive

        # Meta information (not part of the state)
        meta = {
            "timestamp": pygame.time.get_ticks(),
            "outcome": outcome,
            "episode_time": pygame.time.get_ticks() - self.game.ui.start_time,
        }

        # The actual data for learning
        self.human_play_data.append(
            {
                "state": state,  # All observable information
                "action": direction,  # Action taken
                "meta": meta,  # Metadata (not used for learning)
            }
        )

    def get_state(self):
        """
        Returns a state representation for AI training.
        Structure varies based on level for backward compatibility:
        - Level 1: Basic state representation
        - Level 2: Enhanced state with additional fields for advanced training
        """
        # Get the full game grid (2D representation)
        full_grid = []
        for y in range(self.tile_world.height):
            row = []
            for x in range(self.tile_world.width):
                tile = self.tile_world.get_tile(x, y)
                row.append(tile.tile_type if tile else "WALL")
            full_grid.append(row)

        # Get other player position (for all levels)
        other_player_pos = None
        other_player = None
        if self.player_id == 1 and hasattr(self.game, "player2"):
            other_player = self.game.player2
            other_player_pos = [other_player.x, other_player.y]
        elif self.player_id == 2 and hasattr(self.game, "player1"):
            other_player = self.game.player1
            other_player_pos = [other_player.x, other_player.y]

        # Check if we're in level 2
        is_level_2 = (
            hasattr(self.tile_world, "level_index") and self.tile_world.level_index == 1
        )

        from src.agent import RLAgent  # Import here to avoid circular imports

        grid_size = 6 if isinstance(self, RLAgent) else 3

        # Get appropriate local grid with the specified size
        if is_level_2 and hasattr(self.tile_world, "get_augmented_local_grid"):
            # If using augmented grid, we'd need to modify that function too
            local_grid = self.tile_world.get_local_grid(self.x, self.y, size=grid_size)
        else:
            local_grid = self.tile_world.get_local_grid(self.x, self.y, size=grid_size)

        # Get appropriate local grid
        if is_level_2 and hasattr(self.tile_world, "get_augmented_local_grid"):
            # local_grid = self.tile_world.get_augmented_local_grid(self.x, self.y)
            local_grid = self.tile_world.get_local_grid(self.x, self.y)
        else:
            local_grid = self.tile_world.get_local_grid(self.x, self.y)

        # Basic state dictionary that works for any level
        state = {
            # Player state
            "position": [self.x, self.y],
            "player_collected_chips": self.collected_chips,
            "is_sliding": self.is_sliding,
            "is_being_forced": self.is_being_forced,
            "alive": self.alive,
            # Game state
            "full_grid": full_grid,
            "local_grid": local_grid,
            "nearest_chip": self.tile_world.find_nearest_chip(self.x, self.y, "CHIP"),
            "chip_positions": self.tile_world.chip_positions,
            "exit_position": self.tile_world.exit_position,
            "socket_unlocked": self.tile_world.socket_unlocked,
            "total_collected_chips": self.tile_world.collected_chips,
            "remaining_chips": self.tile_world.total_chips
            - self.tile_world.collected_chips,
            # Multiplayer state
            "other_player_position": other_player_pos,
            "player_id": self.player_id,
        }

        # For level 1, use empty dictionaries for keys and boots
        if not is_level_2:
            state["collected_keys"] = {}
            state["collected_boots"] = {}
            return state

        # ==========================================
        # Additional fields for level 2 only below
        # ==========================================

        # Get collected keys/boots
        collected_keys = {k: v for k, v in self.keys.items() if v}
        collected_boots = {k: v for k, v in self.boots.items() if v}
        state["collected_keys"] = collected_keys
        state["collected_boots"] = collected_boots

        nearest_key = self.tile_world.find_nearest_chip(self.x, self.y, "KEY")
        nearest_boot = self.tile_world.find_nearest_chip(self.x, self.y, "BOOT")
        state["nearest_key"] = nearest_key
        state["nearest_boot"] = nearest_boot

        # Get elapsed time
        time_elapsed = 0
        if hasattr(self.game, "start_time"):
            time_elapsed = pygame.time.get_ticks() - self.game.start_time
        elif hasattr(self.game, "clock"):
            # Try to get time, handle both direct attribute and method
            if isinstance(self.game.clock, (int, float)):
                time_elapsed = self.game.clock
            elif hasattr(self.game.clock, "get_time"):
                time_elapsed = self.game.clock.get_time()
        state["time_elapsed"] = time_elapsed

        # Get item assignments
        player_item_assignments = []
        if hasattr(self.game, "ui") and hasattr(self.game.ui, "get_my_assignments"):
            player_item_assignments = self.game.ui.get_my_assignments(
                self, self.player_id, self.x, self.y
            )
        state["player_item_assignments"] = player_item_assignments

        # Get other player data
        other_player_collected_chips = 0
        other_player_collected_keys = {}
        other_player_collected_boots = {}

        if other_player:
            other_player_collected_chips = other_player.collected_chips
            other_player_collected_keys = {
                k: v for k, v in other_player.keys.items() if v
            }
            other_player_collected_boots = {
                k: v for k, v in other_player.boots.items() if v
            }

        state["other_player_collected_chips"] = other_player_collected_chips
        state["other_player_collected_keys"] = other_player_collected_keys
        state["other_player_collected_boots"] = other_player_collected_boots

        # Goal position for layered training
        goal_pos = [-1, -1]  # Default invalid position
        if player_item_assignments and len(player_item_assignments) > 0:
            first_assignment = player_item_assignments[0]
            if "position" in first_assignment:
                goal_pos = first_assignment["position"]
        state["goal_pos"] = goal_pos

        return state

    def remove_self(self):
        # Restore the tile to its original state
        original_tile = self.tile_world.get_tile(self.x, self.y)
        self.tile_world.set_tile(
            self.x,
            self.y,
            Tile(
                original_tile.tile_type,
                original_tile.walkable,
                original_tile.effect,
                self.tile_world.sprite_sheet,
                original_tile.sprite_index,
            ),
        )
        self.alive = False

    def draw(self, screen):
        if self.alive:
            screen.blit(
                self.image,
                (
                    self.x * self.tile_size,
                    (self.y + settings.TOP_UI_SIZE) * self.tile_size,
                ),
            )

    def collision_detection(self, x, y):
        for beetle in self.tile_world.beetles:
            if beetle.x == x and beetle.y == y:
                return True
        return False

    def set_child_attribute(self, value):
        self.is_train = value

    def calculate_path_length(self, start_x, start_y, target_x, target_y):
        """Calculate actual path length considering walls and obstacles."""
        queue = deque([(start_x, start_y, 0)])  # (x, y, steps)
        visited = set([(start_x, start_y)])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Down, Right, Up, Left

        while queue:
            x, y, steps = queue.popleft()

            if x == target_x and y == target_y:
                return steps  # Found the target

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    tile = self.tile_world.get_tile(nx, ny)
                    if tile and tile.walkable:
                        visited.add((nx, ny))
                        queue.append((nx, ny, steps + 1))

        return float("inf")  # No path found

    def get_optimized_assignments(self):
        """Assign collectibles based on actual path length."""
        self.assignments = {}

        for item_pos in self.tile_world.prev_collectable_list:
            # Calculate actual path lengths
            p1_steps = self.calculate_path_length(
                self.game.player1.x, self.game.player1.y, item_pos[0], item_pos[1]
            )

            p2_steps = self.calculate_path_length(
                self.game.player2.x, self.game.player2.y, item_pos[0], item_pos[1]
            )

            # Assign to player with shorter path
            player_id = 1 if p1_steps < p2_steps else 2
            tile = self.tile_world.get_tile(item_pos[0], item_pos[1])
            self.assignments[item_pos] = (player_id, tile.get_sprite())
