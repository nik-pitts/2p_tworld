import pygame

class TileSet:
    def __init__(self, image_path, tile_size):
        self.tile_sheet = pygame.image.load(image_path).convert_alpha()
        self.tile_size = tile_size

    def get_tile(self, col, row):
        """Extracts a tile from the sprite sheet based on column and row index."""
        tile = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)  # Create blank tile
        tile.blit(self.tile_sheet, (0, 0),  # Destination in new surface
                  (col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size))  # Source rect
        tile.set_colorkey((255, 0, 255))

        return tile