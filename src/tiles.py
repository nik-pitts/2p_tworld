from collections import deque
import json
import pygame
from src.tile_definitions import TILE_MAPPING
import src.settings as settings


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
        self.exit_position = None

        # Initialize game state
        self.reset_level_state()
        self.load_level(self.level_index)

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

    def load_level(self, level_index):
        self.reset_level_state()
        """Load a specific level by index."""
        if level_index >= len(self.level_data):
            print("All levels completed! Restarting from Level 1.")
            level_index = 0  # Restart from the first level

        # Fetch level data
        level_data = self.level_data[level_index]
        self.level_index = level_index  # Update current level index
        self.total_chips = level_data["numberOfChips"]
        self.level_time = level_data["time"]
        self.hint = level_data["hintText"]
        self.width = settings.MAP_SIZE  # Dynamically set width
        self.height = settings.MAP_SIZE  # Dynamically set height

        # Reset and initialize the board
        self.board = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.player_positions = []  # Reset player positions
        self.initialize_tiles(level_data["upperLayer"])  # Populate tiles

        print(
            f"Loaded Level {level_index + 1}: {level_data['map_title']} (Chips Required: {self.total_chips}, Time: {self.level_time}s)"
        )

    def initialize_tiles(self, upper_layer):
        """Process and set up tiles based on level data."""
        for y in range(self.height):
            for x in range(self.width):
                tile_id = upper_layer[y * self.width + x]

                if tile_id == 110:
                    self.player_positions.append((x, y))  # Store player start positions
                    tile_id = 0  # Convert player start tile to floor

                if tile_id in TILE_MAPPING:
                    tile_type, walkable, sprite_index, effect = TILE_MAPPING[tile_id]

                    # Store chip positions
                    if tile_type == "CHIP":
                        self.chip_positions.append((x, y))

                    # Store exit position
                    if tile_type == "EXIT":
                        self.exit_position = (x, y)

                    self.set_tile(
                        x,
                        y,
                        Tile(
                            tile_type, walkable, effect, self.sprite_sheet, sprite_index
                        ),
                    )

    def next_level(self):
        """Load the next level"""
        self.load_level(self.level_index + 1)

    def collect_chip(self):
        """Function to collect a chip"""
        self.collected_chips += 1
        print(f"🔹 Chips collected: {self.collected_chips}/{self.total_chips}")
        if self.collected_chips >= self.total_chips:
            self.unlock_socket()

    def unlock_socket(self):
        self.socket_unlocked = True
        print("The socket has been unlocked!")

    def remove_socket(self, x, y):
        print("A player has stepped on the socket! It is now removed.")
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
            if self.collected_chips == self.total_chips:
                return (-1, -1)  # All chips collected, no chips left
            # Find the closest chip using index difference
            nearest_chip = min(
                self.chip_positions,
                key=lambda chip: abs(chip[0] - player_x) + abs(chip[1] - player_y),
            )
            return nearest_chip

    def is_valid_position(self, x, y):
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and self.get_tile(x, y).tile_type != "WALL"
        )

    def draw(self, screen):
        for y in range(self.height):
            for x in range(self.width):
                tile = self.get_tile(x, y)
                sprite = tile.get_sprite()
                if sprite:
                    screen.blit(
                        sprite,
                        (
                            x * self.sprite_sheet.tile_size,
                            y * self.sprite_sheet.tile_size,
                        ),
                    )
