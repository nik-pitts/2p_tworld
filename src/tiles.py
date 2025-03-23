import json
import pygame
from src.tile_definitions import TILE_MAPPING
import src.settings as settings
from src.monster import Beetle


class Tile:
    def __init__(
        self,
        tile_type,
        walkable=None,
        effect=None,
        sprite_sheet=None,
        sprite_index=None,
    ):
        self.tile_type = tile_type
        self.walkable = walkable
        self.effect = effect
        self.sprite_sheet = sprite_sheet
        self.sprite_index = sprite_index

    def get_sprite(self):
        if self.sprite_sheet and self.sprite_index:
            return self.sprite_sheet.get_tile(*self.sprite_index)
        return None


class TileSpriteSheet:
    def __init__(self, image_path, tile_size):
        self.tile_sheet = pygame.image.load(image_path).convert_alpha()
        self.tile_size = tile_size

    def get_tile(self, col, row):
        """Extracts a tile from the sprite sheet based on column and row index."""
        tile = pygame.Surface(
            (self.tile_size, self.tile_size), pygame.SRCALPHA
        )  # Create blank tile
        tile.blit(
            self.tile_sheet,
            (0, 0),  # Destination in new surface
            (
                col * self.tile_size,
                row * self.tile_size,
                self.tile_size,
                self.tile_size,
            ),
        )
        tile.set_colorkey((255, 0, 255))

        return tile


class TileWorld:
    """Class that represents the game world, including the map and player positions."""

    def __init__(self, level_file, sprite_sheet):
        """Initialize TileWorld and load the first level."""
        self.sprite_sheet = sprite_sheet
        self.level_file = level_file  # Store level data file path
        self.level_data = self.load_level_data()  # Load JSON once
        self.level_index = 0  # Track current level
        self.chip_positions = []
        self.key_positions = []
        self.boot_positions = []
        self.exit_position = None

    def load_level_data(self):
        """Load all levels from the JSON file"""
        with open(self.level_file, "r") as f:
            return json.load(f)["levels"]

    def reset_level_state(self):
        """Reset level-related variables before loading a new level."""
        self.total_chips = 0  # Required chips for this level
        self.collected_chips = 0  # Reset collected chips count
        self.socket_unlocked = False  # Reset socket state
        self.level_time = None  # Level time limit
        self.hint = ""  # Reset hint text
        self.player_positions = []  # Reset player positions
        self.width = 0  # Width of the level
        self.height = 0  # Height of the level
        self.board = []  # Reset board
        self.prev_collectable_list = []
        self.collectable_list = []  # Reset collectable list

    def load_level(self, level_index):
        self.reset_level_state()
        """Load a specific level by index."""
        if level_index >= len(self.level_data):
            # print("All levels completed! Restarting from Level 1.")
            level_index = 0  # Restart from the first level

        # Fetch level data
        level_data = self.level_data[level_index]
        self.level_index = level_index  # Update current level index
        self.total_chips = level_data["numberOfChips"]
        self.level_time = level_data["time"]
        self.hint = level_data["hintText"]
        self.lower_layer = level_data["lowerLayer"]
        self.beetle_positions = {}
        self.width = settings.MAP_SIZE  # Dynamically set width
        self.height = settings.MAP_SIZE  # Dynamically set height

        # Reset and initialize the board
        self.board = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.player_positions = []  # Reset player positions
        self.initialize_tiles(level_data["upperLayer"])  # Populate tiles

        # print(
        #     f"Loaded Level {level_index + 1}: {level_data['map_title']} (Chips Required: {self.total_chips}, Time: {self.level_time}s)"
        # )

    def initialize_tiles(self, upper_layer):
        """Process and set up tiles based on level data."""
        self.collectable_list = []
        self.beetles = []  # This will hold dynamic Beetle objects

        for y in range(self.height):
            for x in range(self.width):
                tile_id = upper_layer[y * self.width + x]

                if tile_id == 110:
                    self.player_positions.append((x, y))  # Store player start positions
                    tile_id = 0  # Convert player start tile to floor

                if tile_id in TILE_MAPPING:
                    tile_type, walkable, sprite_index, effect, collectable = (
                        TILE_MAPPING[tile_id]
                    )

                    # Store chip positions
                    if tile_type.startswith("CHIP"):
                        self.chip_positions.append((x, y))

                    if tile_type.startswith("KEY"):
                        self.key_positions.append((x, y))

                    if tile_type.startswith("BOOT"):
                        self.boot_positions.append((x, y))

                    # Store exit position
                    if tile_type == "EXIT":
                        self.exit_position = (x, y)

                    # If this tile represents a beetle...
                    if effect and effect.startswith("BEETLE_"):
                        direction = effect.split("_")[1]  # e.g., "NORTH"
                        print("🪲 Beetle added at", x, y, "moving", direction)

                        # Create a Beetle object and add it to self.beetles
                        beetle = Beetle(x, y, self, direction)
                        self.beetles.append(beetle)

                        # Replace this tile with a FLOOR tile so the static tilemap has no beetle.
                        self.set_tile(
                            x, y, Tile("FLOOR", True, None, self.sprite_sheet, (0, 0))
                        )
                        continue  # Skip normal tile processing

                    # Otherwise, set the tile normally
                    self.set_tile(
                        x,
                        y,
                        Tile(
                            tile_type, walkable, effect, self.sprite_sheet, sprite_index
                        ),
                    )

                if collectable:
                    self.collectable_list.append((x, y))

    def next_level(self):
        """Load the next level"""
        self.load_level(self.level_index + 1)

    def collect_chip(self):
        """Function to collect a chip"""
        self.collected_chips += 1

        # print(f"🔹 Chips collected: {self.collected_chips}/{self.total_chips}")
        if self.collected_chips >= self.total_chips:
            self.unlock_socket()

    def collect_key(self, key_color, player):
        """Track when a player collects a key"""
        # Set the corresponding key in the player's inventory
        player.keys[key_color] = True
        # print(f"🔑 Player {player.player_id} collected a {key_color} key!")

    def unlock_socket(self):
        self.socket_unlocked = True
        # print("The socket has been unlocked!")

    def remove_socket(self, x, y):
        # print("A player has stepped on the socket! It is now removed.")
        self.set_tile(x, y, Tile("FLOOR", True, None, self.sprite_sheet, (0, 0)))

    def set_tile(self, x, y, tile):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.board[y][x] = tile

    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.board[y][x]
        return Tile("WALL", False, None, self.sprite_sheet, (1, 0))

    def find_nearest_chip(self, player_x, player_y, target_type):
        if target_type == "CHIP":
            if not self.chip_positions or self.collected_chips == self.total_chips:
                return (-1, -1)  # All chips collected, no chips left
            # Find the closest chip using index difference
            nearest_chip = min(
                self.chip_positions,
                key=lambda chip: abs(chip[0] - player_x) + abs(chip[1] - player_y),
            )
            return nearest_chip

        elif target_type == "KEY":
            if len(self.key_positions) == 0:
                return (-1, -1)
            # Find the closest chip using index difference
            nearest_key = min(
                self.key_positions,
                key=lambda chip: abs(chip[0] - player_x) + abs(chip[1] - player_y),
            )
            return nearest_key

        elif target_type == "BOOT":
            if len(self.boot_positions) == 0:
                return (-1, -1)
            # Find the closest chip using index difference
            nearest_boot = min(
                self.boot_positions,
                key=lambda chip: abs(chip[0] - player_x) + abs(chip[1] - player_y),
            )
            return nearest_boot

    def is_valid_position(self, x, y):
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and self.get_tile(x, y).tile_type != "WALL"
        )

    def draw(self, screen):
        # draw lower layer
        for tile in self.lower_layer:
            x, y = tile[1], tile[0]
            sprite = self.sprite_sheet.get_tile(0, 0)
            screen.blit(
                sprite,
                (
                    x * self.sprite_sheet.tile_size,
                    (y + settings.TOP_UI_SIZE) * self.sprite_sheet.tile_size,
                ),
            )

        # draw upper layer
        for y in range(self.height):
            for x in range(self.width):
                tile = self.get_tile(x, y)
                sprite = tile.get_sprite()
                if sprite:
                    screen.blit(
                        sprite,
                        (
                            x * self.sprite_sheet.tile_size,
                            (y + settings.TOP_UI_SIZE) * self.sprite_sheet.tile_size,
                        ),
                    )

        for beetle in self.beetles:
            beetle.draw(screen)

    def remove_collectable(self, x, y):
        """Remove a collected item from the collectable_list"""
        if (x, y) in self.collectable_list:
            self.collectable_list.remove((x, y))
            # print(f"Item at ({x}, {y}) removed from collectable list")

    def get_local_grid(self, x, y, size=3):
        """Helper to get local grid view around player."""
        local_grid = []
        radius = size // 2

        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                tile = self.get_tile(x + dx, y + dy)
                row.append(tile.tile_type if tile else "WALL")
            local_grid.append(row)
        return local_grid

    def get_beetle_info(self, x, y):
        # 3. MONSTER AWARENESS - Track positions and directions of nearby beetles
        beetle_positions = []
        beetle_directions = []
        for beetle in self.beetles:
            # Calculate relative position to player (better for learning spatial relationships)
            # rel_x = beetle.x - x
            # rel_y = beetle.y - y
            beetle_positions.append([beetle.x, beetle.y])

            # Encode direction as one-hot
            direction_one_hot = [0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT]
            if beetle.direction == "UP":
                direction_one_hot[0] = 1
            elif beetle.direction == "DOWN":
                direction_one_hot[1] = 1
            elif beetle.direction == "LEFT":
                direction_one_hot[2] = 1
            elif beetle.direction == "RIGHT":
                direction_one_hot[3] = 1
            beetle_directions.append(direction_one_hot)

        # Flatten beetle information
        beetle_info = [
            item for sublist in beetle_positions + beetle_directions for item in sublist
        ]

        return beetle_info

    def get_augmented_local_grid(self, x, y):
        # Enhanced Local Grid with more information
        # One-hot encoding for tile types
        tile_mapping = {
            "WALL": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "FLOOR": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "CHIP": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "EXIT": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "SOCKET": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "WATER": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "FIRE": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "ICE": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "FORCE_FLOOR": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "DOOR": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "KEY": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "MOVABLE_DIRT_BLOCK": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            # Default for unknown tiles
            "DEFAULT": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }

        local_grid_features = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                tile = self.get_tile(nx, ny)

                # Basic tile type
                tile_type = tile.tile_type if tile else "WALL"
                tile_encoding = tile_mapping.get(tile_type, tile_mapping["DEFAULT"])

                # Add special effect information
                if tile and tile.effect:
                    effect_encoding = [
                        0,
                        0,
                        0,
                        0,
                    ]  # [Directional, Hazard, Collectable, Interactive]

                    # Directional effects (force floors, one-ways)
                    if tile.effect.startswith("FORCE_"):
                        effect_encoding[0] = 1
                        # Add direction information
                        if "UP" in tile.effect:
                            effect_encoding.append(1)
                        else:
                            effect_encoding.append(0)
                        if "DOWN" in tile.effect:
                            effect_encoding.append(1)
                        else:
                            effect_encoding.append(0)
                        if "LEFT" in tile.effect:
                            effect_encoding.append(1)
                        else:
                            effect_encoding.append(0)
                        if "RIGHT" in tile.effect:
                            effect_encoding.append(1)
                        else:
                            effect_encoding.append(0)
                    else:
                        effect_encoding.extend([0, 0, 0, 0])

                    # Hazard effects
                    if tile.effect in ["BURN", "DROWN"]:
                        effect_encoding[1] = 1

                    # Collectable effects
                    if tile.effect in [
                        "COLLECT",
                        "RED",
                        "BLUE",
                        "GREEN",
                        "YELLOW",
                        "WATER",
                        "FIRE",
                        "FORCE",
                    ]:
                        effect_encoding[2] = 1

                    # Interactive effects
                    if tile.effect in ["PUSH", "HINT", "UNLOCK"]:
                        effect_encoding[3] = 1
                else:
                    # No effect
                    effect_encoding = [0, 0, 0, 0, 0, 0, 0, 0]

                # Check for beetles at this position
                beetle_here = 0
                for beetle in self.beetles:
                    if beetle.x == nx and beetle.y == ny:
                        beetle_here = 1
                        break

                # Combine all encodings for this tile
                tile_features = tile_encoding + effect_encoding + [beetle_here]
                local_grid_features.extend(tile_features)
        return local_grid_features
