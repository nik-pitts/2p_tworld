import pygame
import src.settings as settings
from random import randint


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

        # Points per player - Each item is worth 10 points
        self.points_p1 = 0
        self.points_p2 = 0

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

        self.game_result = pygame.Rect(
            settings.TILE_SIZE * (settings.MAP_SIZE + settings.TOP_UI_SIZE) // 2 - 150,
            settings.HEIGHT // 2 + 150,
            300,
            50,
        )

        self.op_button = pygame.Rect(
            settings.TILE_SIZE * settings.MAP_SIZE + 20,
            settings.TOP_UI_SIZE * settings.TILE_SIZE + 230,
            240,
            50,
        )

        self.reject_button = pygame.Rect(
            settings.TILE_SIZE * settings.MAP_SIZE // 2 - 250,
            settings.HEIGHT // 2 + 200,
            200,
            50,
        )

        self.accept_button = pygame.Rect(
            settings.TILE_SIZE * settings.MAP_SIZE // 2 + 50,
            settings.HEIGHT // 2 + 200,
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
        self.prev_item_assignments = {}
        self.popup_active = False  # Whether the popup is open
        self.popup_item = None
        self.feedback_settings = {"msg": None, "fbk_init_t": 0, "fbk_dura": 2500}
        self.team_reward = 0

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
        if not self.game.op_time_enabled and not self.game.pause:
            # Only advance the timer during normal gameplay
            elapsed_time = (
                pygame.time.get_ticks() - self.start_time - self.game.op_time_duration
            ) // 1000
        else:
            # In paused states, display the time when the pause began
            if self.game.op_time_enabled:
                # If in OPTIME mode, use OPTIME start as pause point
                elapsed_time = (
                    self.game.op_time_start
                    - self.start_time
                    - self.game.op_time_duration
                ) // 1000
            elif self.game.level_complete:
                elapsed_time = (
                    self.game.level_completed_time
                    - self.start_time
                    - self.game.op_time_duration
                ) // 1000
            else:
                # If game is paused for other reasons, use current frozen time
                elapsed_time = float("inf")

        # Format time display
        time_left = max(0, self.tile_world.level_time - elapsed_time)
        self.team_reward = time_left * 8
        # Format time left display
        seconds = time_left % 60
        formatted_time = f"{seconds:02}"

        self.draw_text_box(screen, "TIME", str(formatted_time), start_y_offset)
        self.draw_text_box(screen, "Team Reward", str(self.team_reward), start_y_offset)

        # Count down time
        if (
            not self.game.op_time_enabled
            and not self.game.pause
            and self.game.touchdown_time_enabled
        ):
            self.show_countdown(screen)

        # Hint
        self.draw_text_box(screen, "HINT", self.hint_text, start_y_offset + 80)

        # Player 1 Inventory
        self.draw_text(screen, "P1 Inventory", start_y_offset + 320)
        self.draw_inventory_box(screen, start_y_offset + 350, self.inventory_p1)

        # Player 2 Inventory
        self.draw_text(screen, "P2 Inventory", start_y_offset + 470)
        self.draw_inventory_box(screen, start_y_offset + 500, self.inventory_p2)

        self.draw_player_points(screen)

        if self.popup_active:
            self.draw_popup_menu(screen)

        # Draw chip identifiers
        self.draw_op_button(screen)

        # Draw REJECT ACCEPT Button
        self.draw_feedback_result(screen)

        # OPTIME
        if self.game.op_time_enabled:
            self.handle_optime_ui(screen)

        # Agent pondering
        if self.game.agent_thinking_settings["is_thinking"]:
            self.draw_agent_thinking(screen)
            if self.feedback_settings["msg"] is not None:
                self.draw_feedback_result(screen)

    def draw_player_points(self, screen):
        """Draw the player points above their sprites"""
        # Player 1 points
        self.draw_points_bubble(
            screen,
            self.points_p1,
            self.player1.x * self.player1.tile_size,
            self.player1.y * self.player1.tile_size
            + settings.TOP_UI_SIZE * settings.TILE_SIZE,
            (0, 0, 0),
        )

        # Player 2 points
        self.draw_points_bubble(
            screen,
            self.points_p2,
            self.player2.x * self.player2.tile_size,
            self.player2.y * self.player2.tile_size
            + settings.TOP_UI_SIZE * settings.TILE_SIZE,
            (0, 0, 0),
        )

    def draw_points_bubble(self, screen, points, x, y, color):
        """Draw a bubble with points value"""
        # Only draw if in the game area (not in UI)
        font = pygame.font.Font("./res/font/MinecraftStandard.ttf", 12)
        text = font.render(str(points), True, (255, 255, 255))

        # Create a circular bubble around the text
        bubble_radius = max(text.get_width() // 2 + 5, 15)

        # Draw bubble
        pygame.draw.circle(screen, color, (x, y), bubble_radius)
        pygame.draw.circle(screen, (255, 255, 255), (x, y), bubble_radius, 1)

        # Draw text centered in bubble
        screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

    def handle_optime_ui(self, screen):
        # Freeze game window
        screen.blit(self.screen_freeze, (0, settings.TOP_UI_SIZE * settings.TILE_SIZE))
        game_board_med = settings.MAP_SIZE * settings.TILE_SIZE // 2
        y_offset = settings.TOP_UI_SIZE * settings.TILE_SIZE
        # Show who requested OPTIME
        if self.game.human.clicked_optime:
            self.draw_optime_text(
                screen, "Player 1 Requested OPTIME", game_board_med - 200, y_offset + 50
            )
        else:
            # Display agent's calling...
            self.draw_optime_text(
                screen, "Player 2 Requested OPTIME", game_board_med - 200, y_offset + 50
            )
            current_time = pygame.time.get_ticks()
            thinking_time_elapsed = current_time - self.game.op_time_start

            if thinking_time_elapsed >= self.game.agent_thinking_settings["delay_time"]:
                # Only clear assignments once when thinking is done
                if not self.game.agent_thinking_settings["is_thinking"]:
                    # copy current item assignment before suggesting
                    self.game.agent_thinking_settings["is_thinking"] = False

                # Update UI with assignments
                for item_pos, assignment in self.game.agent[0].assignments.items():
                    self.item_assignments[item_pos] = assignment

                # Show feedback button
                self.show_feedback_button(screen)
            else:
                # Show thinking animation/text
                thinking_dots = "." * (
                    1 + (thinking_time_elapsed // 500) % 3
                )  # Animated dots
                self.draw_optime_text(
                    screen,
                    f"Thinking{thinking_dots}",
                    game_board_med - 100,
                    y_offset + 200,
                )

    def draw_text_box(self, screen, title, text, y_offset):
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 28)
        font_content = pygame.font.Font("./res/font/MinecraftStandard.ttf", 16)
        title_surface = font.render(title, True, (255, 255, 255))

        if title == "Team Reward":
            screen.blit(
                title_surface, (settings.MAP_SIZE * settings.TILE_SIZE + 110, y_offset)
            )
        else:
            screen.blit(
                title_surface, (settings.MAP_SIZE * settings.TILE_SIZE + 20, y_offset)
            )
        box_width, box_height = 240, 40
        if title == "HINT":
            box_height = 80
        elif title == "TIME":
            box_width = 80
        elif title == "Team Reward":
            box_width = 150

        if title != "Team Reward":
            box_rect = pygame.Rect(
                settings.MAP_SIZE * settings.TILE_SIZE + 20,
                y_offset + 30,
                box_width,
                box_height,
            )
        else:
            box_rect = pygame.Rect(
                settings.MAP_SIZE * settings.TILE_SIZE + 110,
                y_offset + 30,
                box_width,
                box_height,
            )

        pygame.draw.rect(screen, (50, 50, 50), box_rect)

        wrapped_text = self.wrap_text(text, font, box_width - 20)

        for i, line in enumerate(wrapped_text):
            text_surface = font_content.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (box_rect.x + 10, box_rect.y + 4 + (i * 25)))

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
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(
            text_surface, (settings.MAP_SIZE * settings.TILE_SIZE + 20, y_offset)
        )

    def draw_optime_text(self, screen, text, x_pos, y_pos):
        font = pygame.font.Font("./res/font/MinecraftStandard.ttf", 20)
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (x_pos, y_pos))

    def draw_inventory_box(self, screen, y_offset, inventory):
        box_x = settings.MAP_SIZE * settings.TILE_SIZE + 20
        box_y = y_offset
        tile_size = settings.TILE_SIZE

        for row in range(2):
            for col in range(5):
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
            self.points_p1 += 10  # Add 10 points per item
        else:
            self.inventory_p2.append(item)
            self.points_p2 += 10  # Add 10 points per item

        if item_pos in self.item_assignments:
            del self.item_assignments[item_pos]

    def show_hint(self, hint):
        self.hint_text = hint

    def clear_hint(self):
        self.hint_text = ""

    def handle_click(self, event):
        if self.game.pause:
            if self.replay_button.collidepoint(event.pos):
                self.game.restart_game()

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
            if (
                not self.game.op_time_enabled
                and not self.game.agent_thinking_settings["is_thinking"]
            ):
                # when the opbutton is clicked initially copy current item assignments
                self.prev_item_assignments = self.item_assignments.copy()
                self.tile_world.prev_collectable_list = (
                    self.tile_world.collectable_list.copy()
                )
                self.game.op_time_enabled = True
                self.game.human.clicked_optime = True
                self.game.op_time_start = pygame.time.get_ticks()
            else:
                # End OPTIME and trigger agent thinking
                self.game.agent_thinking_settings["init_time"] = pygame.time.get_ticks()
                self.game.agent_thinking_settings["is_thinking"] = True
                self.op_time_text = "OPTIME"

        elif event.button == 1 and self.game.op_time_enabled:
            # reject case
            if self.reject_button.collidepoint(event.pos):
                self.item_assignments = (
                    self.prev_item_assignments.copy()
                    if hasattr(self, "self.prev_item_assignments")
                    else {}
                )
                self.tile_world.collectable_list = (
                    self.tile_world.prev_collectable_list.copy()
                )

                # End OPTIME mode
                self.game.resume_from_optime()
                self.game.op_time_enabled = False
                self.game.human.clicked_optime = False
                self.op_time_text = "OPTIME"

                self.feedback_settings["msg"] = "Plaey 1 Rejected"
                self.feedback_settings["fbk_init_t"] = pygame.time.get_ticks()

            elif self.accept_button.collidepoint(event.pos):
                # End OPTIME mode
                self.game.resume_from_optime()
                self.game.op_time_enabled = False
                self.game.human.clicked_optime = False
                self.op_time_text = "OPTIME"

                # Show rejection feedback
                self.feedback_settings["msg"] = "Plaey 1 Accepted"
                self.feedback_settings["fbk_init_t"] = pygame.time.get_ticks()

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
        screen.blit(text_surface, (self.op_button.x + 55, self.op_button.y + 7))

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

    def show_game_result(self, screen):
        self.show_replay_button(screen)
        self.show_nextlv_button(screen)
        pygame.draw.rect(screen, (0, 0, 0), self.game_result)

        if self.player1.exited and self.player2.exited:
            text = (
                f"Player 1: {str(self.team_reward)}, Player 2: {str(self.team_reward)}"
            )
        elif self.player1.exited and not self.player2.exited:
            text = f"Player 1: {str(self.points_p1)}, Player 2: 0"
        elif self.player2.exited and not self.player1.exited:
            text = f"Player 1: 0, Player 2: {str(self.points_p2)}"
        else:
            text = "Player 1: 0, Player 2: 0"

        font = pygame.font.Font("./res/font/MinecraftStandard.ttf", 14)

        text_surface = font.render(text, True, (255, 255, 255))

        # Center position
        text_x = (
            self.game_result.x
            + (self.game_result.width - text_surface.get_width()) // 2
        )
        text_y = (
            self.game_result.y
            + (self.game_result.height - text_surface.get_height()) // 2
        )

        # Blit text at centered position
        screen.blit(text_surface, (text_x, text_y))

    def show_resume_button(self, screen):
        self.op_time_text = "-SEND-"

    def show_feedback_button(self, screen):
        pygame.draw.rect(screen, (0, 0, 0), self.reject_button)
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 50)
        text_surface = font.render("Reject", True, (255, 255, 255))
        screen.blit(text_surface, (self.reject_button.x + 25, self.reject_button.y + 5))

        pygame.draw.rect(screen, (0, 0, 0), self.accept_button)
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 50)
        text_surface = font.render("Accept", True, (255, 255, 255))
        screen.blit(text_surface, (self.accept_button.x + 25, self.accept_button.y + 5))

    def draw_feedback_result(self, screen):
        current_time = pygame.time.get_ticks()
        msg = self.feedback_settings["msg"]
        feedback_init_time = self.feedback_settings["fbk_init_t"]
        feedback_duraction = self.feedback_settings["fbk_dura"]
        if msg and current_time - feedback_init_time < feedback_duraction:
            font = pygame.font.Font("./res/font/MinecraftStandard.ttf", 18)
            text = font.render(msg, True, (255, 255, 255))

            # Center the text
            text_x = (
                settings.MAP_SIZE * settings.TILE_SIZE
            ) // 2 - text.get_width() // 2
            text_y = settings.TOP_UI_SIZE * settings.TILE_SIZE + 300

            screen.blit(text, (text_x, text_y))
        else:
            # Clear message after duration
            self.feedback_settings["msg"] = None

    def draw_agent_thinking(self, screen):
        current_time = pygame.time.get_ticks()
        thinking_time_elapsed = (
            current_time - self.game.agent_thinking_settings["init_time"]
        )

        if thinking_time_elapsed >= self.game.agent_thinking_settings["delay_time"]:
            # Only clear assignments once when thinking is done
            if self.game.agent_thinking_settings["is_thinking"]:
                decision = self.game.agent[0].alignment
                if decision == 0:  # aligned model direct acception
                    self.feedback_settings["msg"] = (
                        f"Player {self.game.agent[0].player_id} Accepted"
                    )
                    self.feedback_settings["fbk_init_t"] = pygame.time.get_ticks()
                elif decision == 2:  # diverged model direct reject
                    self.item_assignments = self.prev_item_assignments.copy()
                    self.tile_world.collectable_list = (
                        self.tile_world.prev_collectable_list.copy()
                    )
                    self.feedback_settings["msg"] = (
                        f"Player {self.game.agent[0].player_id} Rejected"
                    )
                    self.feedback_settings["fbk_init_t"] = pygame.time.get_ticks()
                elif decision == 1:  # merged model follow up!
                    self.feedback_settings["msg"] = (
                        f"Player {self.game.agent[0].player_id} New Suggestion"
                    )
                    self.feedback_settings["fbk_init_t"] = pygame.time.get_ticks()

            self.game.agent_thinking_settings["is_thinking"] = False
            if decision == 0 or decision == 2:
                self.game.resume_from_optime()
                self.game.human.clicked_optime = False
                self.game.op_time_enabled = False
                self.op_time_text = "OPTIME"
            else:
                self.game.human.clicked_optime = False
                self.game.trigger_optime()

        else:
            # Show thinking animation/text
            thinking_dots = "." * (
                1 + (thinking_time_elapsed // 500) % 3
            )  # Animated dots
            self.draw_optime_text(
                screen,
                f"Thinking{thinking_dots}",
                settings.MAP_SIZE * settings.TILE_SIZE // 2 - 100,
                settings.TOP_UI_SIZE * settings.TILE_SIZE + 200,
            )

    def get_my_assignments(self, player_id, me_x, me_y, max_assignments=8):
        """Get only assignments relevant to this agent."""
        my_assignments = []

        # Filter assignments for this player
        for (x, y), (p_id, _) in self.item_assignments.items():
            if player_id == p_id:
                tile = self.tile_world.get_tile(x, y)
                tile_type = tile.tile_type if tile else "UNKNOWN"

                my_assignments.append(
                    {
                        "position": [x, y],
                        "relative_position": [
                            x - me_x,
                            y - me_y,
                        ],  # Position relative to player
                        "manhattan_distance": abs(x - me_x) + abs(y - me_y),
                        "tile_type": tile_type,
                        "effect": tile.effect if tile and tile.effect else None,
                    }
                )

        # Sort by distance (closest first)
        my_assignments.sort(key=lambda a: a["manhattan_distance"])

        # Pad or truncate to fixed size
        padded = my_assignments[:max_assignments]
        while len(padded) < max_assignments:
            padded.append(
                {
                    "position": [-1, -1],  # Invalid position
                    "relative_position": [0, 0],
                    "manhattan_distance": -1,  # Invalid distance
                    "tile_type": "NONE",
                    "effect": None,
                }
            )

        return padded

    def show_countdown(self, screen):
        font = pygame.font.Font("./res/font/Redpixel-8Mqz2.ttf", 50)
        countdown = 3 - (
            (pygame.time.get_ticks() - self.game.touchdown_time_started) // 1000
        )
        text_surface = font.render(str(countdown), True, (255, 0, 0))
        screen.blit(
            text_surface,
            (
                settings.TILE_SIZE * (settings.MAP_SIZE // 2) + 10,
                settings.TILE_SIZE * (settings.MAP_SIZE + settings.TOP_UI_SIZE) // 2,
            ),
        )
