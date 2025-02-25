import pygame
import src.settings as settings


class GameUI:
    def __init__(self, tile_world, player1, player2, game):
        """Init method for GameUI class."""
        self.tile_world = tile_world
        self.player1 = player1
        self.player2 = player2
        self.game = game

        self.ui_width = settings.UI_SIZE * settings.TILE_SIZE
        self.ui_rect = pygame.Rect(
            settings.MAP_SIZE * settings.TILE_SIZE, 0, self.ui_width, settings.HEIGHT
        )

        self.start_time = pygame.time.get_ticks()
        self.hint_text = ""

        # Inventory per player
        self.inventory_p1 = []
        self.inventory_p2 = []

        self.replay_button = pygame.Rect(
            settings.TILE_SIZE * settings.MAP_SIZE // 2 - 75,
            settings.HEIGHT // 2,
            150,
            50,
        )

    def update_ui(self, screen):
        pygame.font.Font(None, 36)

        # Time
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000
        remaining_time = max(0, self.tile_world.level_time - elapsed_time)
        self.draw_text_box(screen, "TIME", str(remaining_time), 20)

        # Hint
        self.draw_text_box(screen, "HINT", self.hint_text, 100)

        # Player 1 Inventory
        self.draw_text(screen, "PLAYER 1", 280)
        self.draw_inventory_box(screen, 300, self.inventory_p1)

        # Player 2 Inventory
        self.draw_text(screen, "PLAYER 2", 440)
        self.draw_inventory_box(screen, 460, self.inventory_p2)

        if self.player1.exited or self.player2.exited:
            pygame.draw.rect(screen, (0, 0, 0), self.replay_button)
            font = pygame.font.Font(None, 50)
            text_surface = font.render("Replay", True, (255, 255, 255))
            screen.blit(
                text_surface, (self.replay_button.x + 15, self.replay_button.y + 5)
            )

    def draw_text_box(self, screen, title, text, y_offset):
        font = pygame.font.Font(None, 28)
        title_surface = font.render(title, True, (255, 255, 255))
        screen.blit(
            title_surface, (settings.MAP_SIZE * settings.TILE_SIZE + 20, y_offset)
        )

        box_width, box_height = 200, 40
        if title == "HINT":
            box_height = 80

        box_rect = pygame.Rect(
            settings.MAP_SIZE * settings.TILE_SIZE + 20,
            y_offset + 20,
            box_width,
            box_height,
        )
        pygame.draw.rect(screen, (150, 150, 150), box_rect)

        wrapped_text = self.wrap_text(text, font, box_width - 20)

        for i, line in enumerate(wrapped_text):
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (box_rect.x + 10, box_rect.y + 10 + (i * 25)))

    def wrap_text(self, text, font, max_width):
        words = text.split(" ")
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + word + " "
            text_width, _ = font.size(test_line)

            if text_width > max_width:
                lines.append(current_line)
                current_line = word + " "
            else:
                current_line = test_line

        lines.append(current_line)
        return lines

    def draw_text(self, screen, text, y_offset):
        font = pygame.font.Font(None, 28)
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(
            text_surface, (settings.MAP_SIZE * settings.TILE_SIZE + 20, y_offset)
        )

    def draw_inventory_box(self, screen, y_offset, inventory):
        box_x = settings.MAP_SIZE * settings.TILE_SIZE + 20
        box_y = y_offset
        tile_size = settings.TILE_SIZE

        for row in range(2):
            for col in range(4):
                pygame.draw.rect(
                    screen,
                    (150, 150, 150),
                    pygame.Rect(
                        box_x + col * tile_size,
                        box_y + row * tile_size,
                        tile_size,
                        tile_size,
                    ),
                )

        self.draw_inventory_items(screen, box_x, box_y, inventory, tile_size)

    def draw_inventory_items(self, screen, x, y, inventory, tile_size):
        for index, item in enumerate(inventory):
            row, col = divmod(index, 8)
            sprite = item.get_sprite()
            if sprite:
                screen.blit(
                    sprite,
                    (x + col * tile_size, y + row * tile_size),
                )

    def update_inventory(self, player, item):
        if player.player_id == 1:
            self.inventory_p1.append(item)
        else:
            self.inventory_p2.append(item)

    def show_hint(self, hint):
        self.hint_text = hint

    def clear_hint(self):
        self.hint_text = ""

    def handle_click(self, event):
        if self.replay_button.collidepoint(event.pos):
            self.game.restart_game()
