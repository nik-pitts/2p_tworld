import pygame
import src.settings as settings


class Beetle:
    # Define class-level maps
    direction_map = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }
    reverse_map = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
    }

    def __init__(self, x, y, tile_world, direction):
        self.x = x
        self.y = y
        self.tile_world = tile_world
        self.direction = direction  # One of "UP", "DOWN", "LEFT", "RIGHT"
        self.cooldown = 300
        self.last_move_time = pygame.time.get_ticks()

        self.sprite_map = {
            "UP": (4, 0),
            "DOWN": (4, 2),
            "RIGHT": (4, 3),
            "LEFT": (4, 1),
        }
        self.sprite_index = self.sprite_map[self.direction]
        self.image = self.tile_world.sprite_sheet.get_tile(
            *self.sprite_map[self.direction]
        )

    def move(self):
        from src.tiles import Tile  # lazy import to avoid circular issues

        current_time = pygame.time.get_ticks()
        if current_time - self.last_move_time < self.cooldown:
            return  # Not time to move yet

        self.last_move_time = current_time

        # Get movement vector based on current direction
        dx, dy = Beetle.direction_map[self.direction]
        next_x, next_y = self.x + dx, self.y + dy
        next_tile = self.tile_world.get_tile(next_x, next_y)

        if next_tile.tile_type == "FLOOR":
            # Replace the current beetle tile with a FLOOR tile
            floor_tile = Tile("FLOOR", True, None, self.tile_world.sprite_sheet, (0, 0))
            self.tile_world.set_tile(self.x, self.y, floor_tile)

            # Update beetle's position
            self.x, self.y = next_x, next_y

            # Set the new position to a beetle tile
            beetle_type = f"BEETLE_{self.direction}"
            beetle_tile = Tile(
                beetle_type,
                False,
                beetle_type,
                self.tile_world.sprite_sheet,
                self.sprite_map[self.direction],
            )
            self.tile_world.set_tile(self.x, self.y, beetle_tile)

            # Update the beetle's image (so it draws correctly)
            self.image = self.tile_world.sprite_sheet.get_tile(
                *self.sprite_map[self.direction]
            )
        else:
            # If the next tile is blocked, change direction
            self.change_direction()

    def change_direction(self):
        new_direction = Beetle.reverse_map[self.direction]
        self.direction = new_direction
        self.sprite_index = self.sprite_map[self.direction]
        self.image = self.tile_world.sprite_sheet.get_tile(
            *self.sprite_map[self.direction]
        )

    def draw(self, screen):
        # First, draw the floor tile at the beetle's position.
        floor_tile = self.tile_world.sprite_sheet.get_tile(0, 0)
        screen.blit(
            floor_tile,
            (
                self.x * settings.TILE_SIZE,
                (self.y + settings.TOP_UI_SIZE) * settings.TILE_SIZE,
            ),
        )
        # Then, draw the beetle sprite on top.
        screen.blit(
            self.image,
            (
                self.x * settings.TILE_SIZE,
                (self.y + settings.TOP_UI_SIZE) * settings.TILE_SIZE,
            ),
        )
