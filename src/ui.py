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
        self.op_time_text = "OPTIME"

        # Inventory per player
        self.inventory_p1 = []
        self.inventory_p2 = []

        self.replay_button = pygame.Rect(
            settings.TILE_SIZE * (settings.MAP_SIZE + settings.TOP_UI_SIZE) // 2 - 150,
            settings.HEIGHT // 2 + 50,
            300,
            50,
        )

        self.nextlv_button = pygame.Rect(
            settings.TILE_SIZE * (settings.MAP_SIZE + settings.TOP_UI_SIZE) // 2 - 150,
            settings.HEIGHT // 2 - 50,
            300,
            50,
        )

        self.op_button = pygame.Rect(
            settings.TILE_SIZE * settings.MAP_SIZE + 20,
            settings.TOP_UI_SIZE * settings.TILE_SIZE + 230,
            200,
            50,
        )

        self.screen_freeze = pygame.Surface(
            (
                settings.MAP_SIZE * settings.TILE_SIZE,
                settings.MAP_SIZE * settings.TILE_SIZE,
            ),
            pygame.SRCALPHA,
        )
        self.screen_freeze.fill((255, 255, 255, 128))

        # Communication Protocol
        self.item_assignments = {}  # {item_pos: assigned_player_id}
        self.popup_active = False  # Whether the popup is open
        self.popup_item = None

    def update_ui(self, screen):
        pygame.font.Font(None, 36)

        title_font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 36)
        wrapped_title = self.wrap_text(
            "Little Cooperative Machines 1.0",
            title_font,
            settings.TILE_SIZE * settings.UI_SIZE,
        )
        for i, line in enumerate(wrapped_title):
            text_surface = title_font.render(line, True, (255, 0, 0))
            screen.blit(
                text_surface,
                (settings.MAP_SIZE * settings.TILE_SIZE + 20, 50 + (i * 60)),
            )

        self.draw_collectable_items(screen)
        self.draw_assignment_bar(screen)

        start_y_offset = settings.TOP_UI_SIZE * settings.TILE_SIZE

        # Time
        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000
        remaining_time = max(0, self.tile_world.level_time - elapsed_time)
        self.draw_text_box(screen, "TIME", str(remaining_time), start_y_offset)

        # Hint
        self.draw_text_box(screen, "HINT", self.hint_text, start_y_offset + 80)

        # Player 1 Inventory
        self.draw_text(screen, "P1 Inventory", start_y_offset + 320)
        self.draw_inventory_box(screen, start_y_offset + 350, self.inventory_p1)

        # Player 2 Inventory
        self.draw_text(screen, "P2 Inventory", start_y_offset + 470)
        self.draw_inventory_box(screen, start_y_offset + 500, self.inventory_p2)

        if self.popup_active:
            self.draw_popup_menu(screen)

        # Draw chip identifiers
        self.draw_op_button(screen)

        #
        if self.game.op_time_enabled:
            screen.blit(
                self.screen_freeze, (0, settings.TOP_UI_SIZE * settings.TILE_SIZE)
            )

    def draw_text_box(self, screen, title, text, y_offset):
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 28)
        title_surface = font.render(title, True, (255, 0, 0))
        screen.blit(
            title_surface, (settings.MAP_SIZE * settings.TILE_SIZE + 20, y_offset)
        )

        box_width, box_height = 200, 40
        if title == "HINT":
            box_height = 80

        box_rect = pygame.Rect(
            settings.MAP_SIZE * settings.TILE_SIZE + 20,
            y_offset + 30,
            box_width,
            box_height,
        )
        pygame.draw.rect(screen, (50, 50, 50), box_rect)

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
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 28)
        text_surface = font.render(text, True, (255, 0, 0))
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
                    (50, 50, 50),
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
            row, col = divmod(index, 4)  # 4 items per row
            sprite = item.get_sprite()
            if not sprite:
                continue

            item_pos = (x + col * tile_size, y + row * tile_size)
            screen.blit(sprite, item_pos)

    def update_inventory(self, player, item, item_pos):
        if player.player_id == 1:
            self.inventory_p1.append(item)
        else:
            self.inventory_p2.append(item)

        if item_pos in self.item_assignments:
            del self.item_assignments[item_pos]

    def show_hint(self, hint):
        self.hint_text = hint

    def clear_hint(self):
        self.hint_text = ""

    def handle_click(self, event):
        if self.game.pause:
            if self.replay_button.collidepoint(event.pos):
                self.game.load_game(next_level=False)

            if self.nextlv_button.collidepoint(event.pos):
                self.game.load_game(next_level=True)

        if event.button == 3:  # Right-click
            x, y = event.pos

            # Check if clicked on collectable items
            item_clicked = self.get_clicked_collectable_item(x, y)
            if item_clicked:
                self.popup_active = True  # Show popup
                self.popup_item = item_clicked
                self.popup_position = (x, y)
                return

            # Check if clicked on assigned items
            assigned_item_clicked = self.get_clicked_assigned_item(x, y)
            if assigned_item_clicked:
                self.popup_active = True  # Show unassign option
                self.popup_item = assigned_item_clicked
                self.popup_position = (x, y)
                return

        elif event.button == 1 and self.popup_active:  # Left-click when popup is open
            self.handle_popup_selection(event.pos)
            self.popup_active = False  # Hide popup after selection

        elif event.button == 1 and self.op_button.collidepoint(event.pos):
            if not self.game.op_time_enabled:
                self.game.op_time_enabled = True
            else:
                self.game.op_time_enabled = False

    def get_clicked_collectable_item(self, x, y):
        """Checks if a collectable item was clicked."""
        item_spacing = settings.TILE_SIZE
        x_offset = 20  # Same as in `draw_collectable_items`
        y_offset = 40

        for item_pos in self.tile_world.collectable_list:
            tile = self.tile_world.get_tile(item_pos[0], item_pos[1])
            sprite = tile.get_sprite()
            if sprite:
                rect = pygame.Rect(
                    x_offset, y_offset, settings.TILE_SIZE, settings.TILE_SIZE
                )
                if rect.collidepoint(x, y):
                    return item_pos  # Return the position of the clicked item
                x_offset += item_spacing  # Move to the next item

        return None  # No item clicked

    def get_clicked_assigned_item(self, x, y):
        """Checks if an assigned item was clicked."""
        box_x = 20
        box_y = 150  # Assignment box y-position
        box_width = (settings.TILE_SIZE * settings.MAP_SIZE - box_x - 20) // 2

        # Create lists to track positions for each player
        p1_positions = []
        p2_positions = []

        # First, calculate all the positions
        x_offset_p1 = box_x + 4
        x_offset_p2 = box_x + box_width + 24

        for item_pos, (player_id, _) in self.item_assignments.items():
            if player_id == 1:
                p1_positions.append((item_pos, x_offset_p1))
                x_offset_p1 += settings.TILE_SIZE
            else:
                p2_positions.append((item_pos, x_offset_p2))
                x_offset_p2 += settings.TILE_SIZE

        # Now check if any position was clicked
        for item_pos, offset_x in p1_positions:
            rect = pygame.Rect(
                offset_x, box_y + 4, settings.TILE_SIZE, settings.TILE_SIZE
            )
            if rect.collidepoint(x, y):
                return item_pos

        for item_pos, offset_x in p2_positions:
            rect = pygame.Rect(
                offset_x, box_y + 4, settings.TILE_SIZE, settings.TILE_SIZE
            )
            if rect.collidepoint(x, y):
                return item_pos

        return None

    def draw_popup_menu(self, screen):
        """Draws the right-click popup menu for assigning or unassigning items."""
        font = pygame.font.Font(None, 28)

        if self.popup_item in self.tile_world.collectable_list:
            options = ["Assign to Player 1", "Assign to Player 2", "Cancel"]
        else:
            options = ["Unassign Item", "Cancel"]

        popup_width = 180
        popup_height = 30 * len(options)
        popup_x, popup_y = self.popup_position

        pygame.draw.rect(
            screen, (50, 50, 50), (popup_x, popup_y, popup_width, popup_height)
        )
        pygame.draw.rect(
            screen, (255, 255, 255), (popup_x, popup_y, popup_width, popup_height), 2
        )

        for i, option in enumerate(options):
            option_surface = font.render(option, True, (255, 255, 255))
            screen.blit(option_surface, (popup_x + 10, popup_y + 10 + (i * 30)))

    def handle_popup_click(self, event):
        """Handles clicks inside the popup menu."""
        if not self.popup_active:
            return

        menu_x, menu_y = pygame.mouse.get_pos()
        if menu_x <= event.pos[0] <= menu_x + 120:
            if menu_y <= event.pos[1] <= menu_y + 30:
                self.item_assignments[self.popup_item] = (
                    self.player1.player_id
                )  # Assign to self
            elif menu_y + 30 <= event.pos[1] <= menu_y + 60:
                self.item_assignments[self.popup_item] = (
                    self.player2.player_id
                )  # Assign to partner

        self.popup_active = False  # Close popup
        self.popup_item = None  # Reset selection

    def handle_popup_selection(self, position):
        """Handles selection inside the popup menu."""
        popup_x, popup_y = self.popup_position
        option_height = 30
        selected_index = (position[1] - popup_y) // option_height

        if self.popup_item in self.tile_world.collectable_list:
            tile = self.tile_world.get_tile(self.popup_item[0], self.popup_item[1])
            sprite = tile.get_sprite()

            if selected_index == 0:  # Assign to Player 1
                self.item_assignments[self.popup_item] = (1, sprite)
                self.tile_world.collectable_list.remove(self.popup_item)
            elif selected_index == 1:  # Assign to Player 2
                self.item_assignments[self.popup_item] = (2, sprite)
                self.tile_world.collectable_list.remove(self.popup_item)

        elif self.popup_item in self.item_assignments:
            if selected_index == 0:  # Unassign Item
                self.tile_world.collectable_list.append(self.popup_item)
                del self.item_assignments[self.popup_item]

        self.popup_active = False  # Hide popup

    def draw_collectable_items(self, screen):
        """Draws all collectable items inside a UI box at the top of the game screen."""
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 32)

        # Draw title "Items:"
        title_surface = font.render("ITEM", True, (255, 255, 255))
        screen.blit(title_surface, (20, 20))  # Positioning at top left

        # Box dimensions
        box_x = 20
        box_y = 50  # Below the title
        box_width = settings.TILE_SIZE * settings.MAP_SIZE - box_x
        box_height = settings.TILE_SIZE + 8  # Single row for items

        # Draw the item box
        pygame.draw.rect(screen, (50, 50, 50), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(screen, (0, 0, 0), (box_x, box_y, box_width, box_height), 2)

        # Item rendering inside the box
        item_spacing = settings.TILE_SIZE
        x_offset = box_x + 4  # Start inside the box
        y_offset = box_y + 4  # Center items vertically in the box

        for item_pos in self.tile_world.collectable_list:
            if item_pos in self.item_assignments:  # ✅ Skip assigned items
                continue

            tile = self.tile_world.get_tile(item_pos[0], item_pos[1])
            sprite = tile.get_sprite()

            if sprite:
                screen.blit(sprite, (x_offset, y_offset))
                x_offset += item_spacing  # Space between items

    def draw_assignment_bar(self, screen):
        # Box dimensions
        box_x = 20
        box_y = 150  # Below the title
        box_width = (settings.TILE_SIZE * settings.MAP_SIZE - box_x - 20) // 2
        box_height = settings.TILE_SIZE + 8  # Single row for items

        """Draws the assignment bar below the collectable item bar."""
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 32)
        p1_surface = font.render("PLAYER 1", True, (255, 255, 255))
        screen.blit(p1_surface, (20, 120))  # Position title under collectable items
        p2_surface = font.render("PLAYER 2", True, (255, 255, 255))
        screen.blit(
            p2_surface, (box_x + box_width + 20, 120)
        )  # Position title under collectable items

        # Draw the item box player 1 inventory
        pygame.draw.rect(screen, (50, 50, 50), (box_x, box_y, box_width, box_height))
        pygame.draw.rect(screen, (0, 0, 0), (box_x, box_y, box_width, box_height), 2)

        # Draw the item box player 2 inventory
        pygame.draw.rect(
            screen, (50, 50, 50), (box_x + box_width + 20, box_y, box_width, box_height)
        )
        pygame.draw.rect(
            screen, (0, 0, 0), (box_x + box_width + 20, box_y, box_width, box_height), 2
        )

        # Render assigned items inside the assignment bar
        item_spacing = settings.TILE_SIZE
        x_offset_p1 = box_x + 4  # Start inside the box
        x_offset_p2 = box_x + box_width + 24  # Start inside the box
        y_offset = box_y + 4  # Center items vertically in the box

        for item_pos, (player_id, sprite) in self.item_assignments.items():
            if sprite:
                # Scale sprite to half size
                scaled_sprite = pygame.transform.scale(
                    sprite, (settings.TILE_SIZE, settings.TILE_SIZE)
                )

                if player_id == 1:  # 🔹 Player 1 Assignment
                    screen.blit(scaled_sprite, (x_offset_p1, y_offset))
                    x_offset_p1 += item_spacing  # Move right for next item
                else:  # 🔹 Player 2 Assignment
                    screen.blit(scaled_sprite, (x_offset_p2, y_offset))
                    x_offset_p2 += item_spacing  # Move right for next item

    def draw_op_button(self, screen):
        pygame.draw.rect(screen, (50, 50, 50), self.op_button)
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 40)
        text_surface = font.render(self.op_time_text, True, (255, 255, 255))
        screen.blit(text_surface, (self.op_button.x + 30, self.op_button.y + 7))

    def show_replay_button(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), self.replay_button)
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 50)
        text_surface = font.render("Replay", True, (255, 255, 255))
        screen.blit(text_surface, (self.replay_button.x + 80, self.replay_button.y + 5))

    def show_nextlv_button(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), self.nextlv_button)
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 50)
        text_surface = font.render("Next Level", True, (255, 255, 255))
        screen.blit(text_surface, (self.nextlv_button.x + 25, self.nextlv_button.y + 5))

    def show_resume_button(self, screen):
        self.op_time_text = "RESUME"
