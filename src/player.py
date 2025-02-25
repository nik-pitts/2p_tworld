import pygame
from src.tiles import Tile
import src.settings as settings


class Player:
    def __init__(self, x, y, tile_world, player_id, record=False):
        self.x, self.y = x, y
        self.tile_size = settings.TILE_SIZE
        self.direction = "DOWN"
        self.speed = 1
        self.tile_world = tile_world
        self.player_id = player_id
        self.exited = False
        self.collected_chips = 0
        self.human_play_data = []
        self.record = record

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

        # Apply tile effects
        if tile.effect == "BURN":
            print(f"Player {self.player_id} burned!")
        elif tile.effect == "DROWN":
            print(f"Player {self.player_id} drowned!")
        elif tile.effect == "SLIDE":
            print(f"Player {self.player_id} slid on ice!")

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
            self.tile_world.game_ui.update_inventory(self, tile)

        # Exit handling
        if tile.effect == "EXIT" and self.tile_world.socket_unlocked:
            print(f"Player {self.player_id} escaped!")
            self.exited = True

        # Wall collision detection
        if tile.walkable:
            self.x, self.y = new_x, new_y

        self.image = self.sprites[self.direction]

    def move(self, event):
        if self.exited:
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

    def draw(self, screen):
        screen.blit(
            self.image,
            (self.x * self.tile_size, (self.y + settings.TOP_UI_SIZE) * self.tile_size),
        )
