import pygame
import time
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
        self.sliding = False
        self.slide_dx, self.slide_dy = 0, 0

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
            },
            2: {
                "UP": pygame.K_UP,
                "DOWN": pygame.K_DOWN,
                "LEFT": pygame.K_LEFT,
                "RIGHT": pygame.K_RIGHT,
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
        if not self.alive or self.exited or self.sliding:
            return  # Ignore movement if dead, exited, or sliding

        new_x, new_y = self.x + dx, self.y + dy
        tile = self.tile_world.get_tile(new_x, new_y)

        # Remove hint when moving
        self.tile_world.game_ui.clear_hint()

        # Socket tile handling
        if tile.tile_type == "SOCKET" and not self.tile_world.socket_unlocked:
            print(f"Player {self.player_id} cannot pass the socket yet!")
            return
        if tile.tile_type == "SOCKET" and self.tile_world.socket_unlocked:
            print(f"Player {self.player_id} stepped on the socket! Removing it.")
            self.tile_world.remove_socket(new_x, new_y)
            self.x, self.y = new_x, new_y

        # **PLAYER DEATH CONDITIONS**
        if tile.effect in ["BURN", "DROWN"]:
            # Check if player has corresponding boots
            if (tile.effect == "BURN" and self.boots["FIRE"]) or (
                tile.effect == "DROWN" and self.boots["WATER"]
            ):
                # Player has appropriate boots, just move normally
                print(
                    f"Player {self.player_id}'s boots protected them from {tile.effect.lower()}ing!"
                )
                self.x, self.y = new_x, new_y
                self.log_move(self.direction)
                self.image = self.sprites[self.direction]
            else:
                # Player doesn't have boots, they die
                print(f"Player {self.player_id} {tile.effect.lower()}ed!")
                self.remove_self()
                self.game.check_game_over()
            return

        # **SLIDE MECHANIC**
        if tile.effect == "SLIDE":
            self.slide_on_ice(dx, dy)
            # if self.boots["WATER"]:  # Water boots also work on ice
            #     # Player has water boots, just move normally
            #     print(
            #         f"Player {self.player_id}'s water boots prevented them from sliding on ice!"
            #     )
            #     self.x, self.y = new_x, new_y
            #     self.log_move(self.direction)
            #     self.image = self.sprites[self.direction]
            # else:
            #     # Player doesn't have water boots, they slide
            #     print(f"Player {self.player_id} slid on ice!")
            #     self.slide_on_ice(dx, dy)
            return

        # **FORCE MECHANIC**
        if tile.effect and tile.effect.startswith("FORCE_"):
            if self.boots["FORCE"]:
                # Player has force boots, just move normally
                print(
                    f"Player {self.player_id}'s force boots prevented them from being moved!"
                )
                self.x, self.y = new_x, new_y
                self.log_move(self.direction)
                self.image = self.sprites[self.direction]
            else:
                # Player doesn't have force boots, they get moved
                print(f"Player {self.player_id} stepped on a force floor!")
                self.x, self.y = new_x, new_y
                self.image = self.sprites[self.direction]
                self.force_by_floor(tile.effect)
            return

        # Hint tile handling
        if tile.effect == "HINT":
            print(f"Player {self.player_id} found a hint: {self.tile_world.hint}")
            self.tile_world.game_ui.show_hint(self.tile_world.hint)

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
                    sprite_sheet=self.tile_world.sprite_sheet,
                    sprite_index=(0, 0),
                ),
            )

            self.tile_world.remove_collectable(new_x, new_y)
            collected_position = (new_x, new_y)  # Get the position
            self.tile_world.game_ui.update_inventory(self, tile, collected_position)

        # Key collection
        if tile.tile_type == "KEY":
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
            # Update UI inventory
            self.tile_world.remove_collectable(new_x, new_y)
            collected_position = (new_x, new_y)
            self.tile_world.game_ui.update_inventory(self, tile, collected_position)

        # Boot collection
        if tile.tile_type == "BOOT":
            boot_type = tile.effect  # This will be "WATER", "FIRE", or "FORCE"
            self.boots[boot_type] = True
            print(f"👢 Player {self.player_id} collected {boot_type} boots!")
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
            # Update UI inventory
            self.tile_world.remove_collectable(new_x, new_y)
            collected_position = (new_x, new_y)
            self.tile_world.game_ui.update_inventory(self, tile, collected_position)

        # Door handling - add this before the "if tile.walkable:" check
        if tile.tile_type == "DOOR":
            door_color = tile.effect  # This should match the key color required
            if self.keys.get(door_color, False):
                print(f"Player {self.player_id} unlocked a {door_color} door!")
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
                self.log_move(self.direction)
                self.image = self.sprites[self.direction]
                return
            else:
                print(
                    f"Player {self.player_id} needs a {door_color} key to unlock this door!"
                )
                return

        if tile.effect == "PUSH":
            self.push_block(new_x, new_y, dx, dy)

        # Exit handling
        if tile.effect == "EXIT" and self.tile_world.socket_unlocked:
            print(f"Player {self.player_id} escaped!")
            self.exited = True

        # Wall collision detection
        if tile.walkable:
            self.x, self.y = new_x, new_y
            self.log_move(self.direction)

        self.image = self.sprites[self.direction]

    def slide_on_ice(self, dx, dy):
        """Prepares the player to slide on ice continuously until hitting a non-slide tile."""
        # Set up sliding state
        self.is_sliding = True
        self.slide_movement_queue = []

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
                print(f"Player {self.player_id} {effect.lower()}ed while sliding!")
                self.remove_self()
                self.game.check_game_over()
                self.is_sliding = False
                return True

            # Move player
            self.x, self.y = next_x, next_y

            # Check if sliding is complete
            if not self.slide_movement_queue:
                self.is_sliding = False

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
            else:
                break

        # Start force movement animation clock
        self.force_move_time = pygame.time.get_ticks()
        self.force_move_delay = 80  # milliseconds between steps

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
                print(f"Player {self.player_id} {effect.lower()}ed while being forced!")
                self.remove_self()
                self.game.check_game_over()
                self.is_being_forced = False
                return True

            # Move player to next position
            self.x, self.y = next_x, next_y

            # If queue is empty, end forced movement
            if not self.force_movement_queue:
                self.is_being_forced = False

            return True

        return False

    def push_block(self, block_x, block_y, dx, dy):
        """Handles pushing a movable block."""
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

        if self.record:
            self.log_move(self.direction)

        self.process_move(dx, dy)

    def log_move(self, direction):
        state = self.get_state()
        self.human_play_data.append({"state": state, "action": direction})

    def get_state(self):
        """Returns a structured state representation for AI training."""

        # 3x3 Local Grid (Surrounding Tiles)
        local_grid = []
        for dy in range(-2, 3):
            row = []
            for dx in range(-2, 3):
                tile = self.tile_world.get_tile(self.x + dx, self.y + dy)
                row.append(tile.tile_type if tile else "WALL")
            local_grid.append(row)

        # Nearest Chip Calculation
        nearest_chip = self.tile_world.find_nearest_chip(self.x, self.y, "CHIP")

        return {
            "position": [self.x, self.y],
            "chips_collected": self.collected_chips,
            "total_collected_chips": self.tile_world.collected_chips,
            "local_grid": local_grid,
            "nearest_chip": nearest_chip,
            "exit_position": self.tile_world.exit_position,
            "socket_unlocked": self.tile_world.socket_unlocked,
        }

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
