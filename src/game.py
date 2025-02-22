import math

import pygame
from src.player import Player
from src.tiles import TileSet
import src.settings as settings

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        # Load tile set
        self.tile_set = TileSet(settings.TILE_SHEET_PATH, settings.TILE_SIZE)

        # Load floor tile
        self.floor_tile = self.tile_set.get_tile(0, 0)

        # Initialize Player (Using tile sheet path)
        self.player = Player(math.floor(settings.MAP_SIZE/2) * settings.TILE_SIZE,
                             math.floor(settings.MAP_SIZE/2) * settings.TILE_SIZE,
                             settings.TILE_SHEET_PATH)

    def draw_grid(self):
        """Draws the tile grid on the screen."""
        for row in range(settings.MAP_SIZE):
            for col in range(settings.MAP_SIZE):
                self.screen.blit(self.floor_tile, (col * settings.TILE_SIZE, row * settings.TILE_SIZE))

    def run(self):
        """Main game loop."""
        while self.running:
            self.screen.fill((0, 0, 0))  # Clear screen
            self.draw_grid()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # player movement
                if event.type == pygame.KEYDOWN:
                    self.player.move(event.key)

            self.player.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(settings.FPS)

        pygame.quit()