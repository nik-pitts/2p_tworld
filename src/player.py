import pygame
from src.tiles import TileSet  # Import the TileSet class
import src.settings as settings

class Player:
    def __init__(self, x, y, tile_sheet_path):
        """Initialize player with directional sprites from TileSet."""
        self.x, self.y = x, y
        self.tile_size = settings.TILE_SIZE
        self.direction = "DOWN"  # Default direction
        self.speed = settings.TILE_SIZE  # Grid-based movement

        # Create a TileSet instance
        self.tile_set = TileSet(tile_sheet_path, self.tile_size)

        # Load directional sprites from tile sheet
        self.sprites = {
            "UP": self.tile_set.get_tile(6, 12),
            "LEFT": self.tile_set.get_tile(6, 13),
            "DOWN": self.tile_set.get_tile(6, 14),
            "RIGHT": self.tile_set.get_tile(6, 15),
        }

        self.image = self.sprites["DOWN"]  # Default sprite

    def move(self, keys):
        """Move player based on key input and update sprite direction."""
        if keys == pygame.K_w:  # Move UP
            self.y -= self.speed
            self.direction = "UP"
        elif keys == pygame.K_s:  # Move DOWN
            self.y += self.speed
            self.direction = "DOWN"
        elif keys == pygame.K_a:  # Move LEFT
            self.x -= self.speed
            self.direction = "LEFT"
        elif keys == pygame.K_d:  # Move RIGHT
            self.x += self.speed
            self.direction = "RIGHT"

        # Update the sprite image
        self.image = self.sprites[self.direction]

    def draw(self, screen):
        """Draw the player at its current position."""
        screen.blit(self.image, (self.x, self.y))