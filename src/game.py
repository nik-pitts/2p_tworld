import pygame
import src.settings as settings
from src.player import Player
from src.tiles import TileSpriteSheet, TileWorld
from src.ui import GameUI
from src.agent import RuleBasedAgent, TreeBasedAgent, BehaviorClonedAgent
from src.data_utils import save_human_data


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.pause = False
        self.op_time_enabled = False

        # Load background once instead of every frame
        self.background_image = pygame.image.load("./res/backgroundimg.png")

        # Load tile sprite sheet
        self.tile_sprite_sheet = TileSpriteSheet(
            settings.TILE_SHEET_PATH, settings.TILE_SIZE
        )
        self.tile_world = TileWorld(settings.LEVEL_DATA_PATH, self.tile_sprite_sheet)

        # Load game
        self.load_game(next_level=False)

    def load_game(self, next_level=False):
        """Loads a new level or restarts the game."""
        if next_level:
            self.tile_world.level_index += 1
        else:
            self.tile_world.level_index = self.tile_world.level_index

        self.pause = False
        self.op_time_enabled = False
        self.tile_world.load_level(self.tile_world.level_index)

        # Load player positions
        player_positions = self.tile_world.player_positions
        if len(player_positions) < 2:
            raise ValueError(
                "At least two player positions must be defined in the level data!"
            )

        # Create players
        self.player1 = Player(
            player_positions[0][0],
            player_positions[0][1],
            self.tile_world,
            self,
            1,
            record=False,
        )

        # self.player2 = Player(
        #     player_positions[1][0],
        #     player_positions[1][1],
        #     self.tile_world,
        #     self,
        #     2,
        #     record=False,
        # )

        self.player2 = BehaviorClonedAgent(
            player_positions[1][0],
            player_positions[1][1],
            self.tile_world,
            self,
            2,
            "./model/lv1_bc_model_6.0.pth",
        )

        # Initialize UI
        self.ui = GameUI(self.tile_world, self.player1, self.player2, self)
        self.tile_world.game_ui = self.ui  # Allow TileWorld to reference UI

    def check_level_complete(self):
        """Checks if both players have reached the exit."""
        if self.player1.exited or self.player2.exited:
            print(f"🎉 Level {self.tile_world.level_index + 1} Complete!")
            self.pause = True
            self.ui.show_replay_button(self.screen)
            self.ui.show_nextlv_button(self.screen)

    def restart_game(self):
        """Restarts the game after game over."""
        print("Restarting game...")
        save_human_data(self.player2)
        self.running = True
        self.load_game()

    def check_game_over(self):
        """Ends the game when both players are dead."""
        if not (self.player1.alive and self.player2.alive):
            print("💀💀 Both players are dead! Game over.")
            self.pause = True  # Pause game loop and wait for event
            self.ui.show_replay_button(self.screen)
            self.ui.show_nextlv_button(self.screen)

    def run(self):
        """Main game loop."""
        while self.running:
            self.screen.blit(self.background_image, (0, 0))

            # Always handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    save_human_data(self.player2)
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print(f"Mouse Click at {event.pos}")
                    self.ui.handle_click(event)

                # Block player input when paused or being forced
                if (
                    not self.pause
                    and not self.player1.is_being_forced
                    and not self.op_time_enabled
                ):
                    if event.type == pygame.KEYDOWN:
                        self.player1.move(event)

            # Always draw the tile world
            self.tile_world.draw(self.screen)

            # Always draw players (Even when paused!)
            self.player1.draw(self.screen)
            self.player2.draw(self.screen)

            if not self.pause and not self.op_time_enabled:
                # Check for and update player movement animations
                player1_animated = (
                    self.player1.update_forced_movement()
                    or self.player1.update_sliding_movement()
                )

                player2_animated = (
                    self.player2.update_forced_movement()
                    or self.player2.update_sliding_movement()
                )

                # Check collisions
                if self.player1.collision_detection(self.player1.x, self.player1.y):
                    self.player1.remove_self()
                    self.check_game_over()

                # Player 2's turn (only if it's an agent and no animations are playing)
                if (
                    not player1_animated
                    and not player2_animated
                    and isinstance(
                        self.player2,
                        (RuleBasedAgent, TreeBasedAgent, BehaviorClonedAgent),
                    )
                ):
                    self.player2.step()

                # Move monsters only if game is running and no animations are playing
                if not player1_animated and not player2_animated:
                    for beetle in self.tile_world.beetles:
                        beetle.move()

                # Check for level completion
                self.check_level_complete()
            elif not self.op_time_enabled:
                # Show Replay and Next Level buttons when paused
                self.ui.show_replay_button(self.screen)
                self.ui.show_nextlv_button(self.screen)
            else:
                self.ui.show_resume_button(self.screen)

            # Always update UI
            self.ui.update_ui(self.screen)
            pygame.display.flip()
            self.clock.tick(settings.FPS)

        save_human_data(self.player1)
        pygame.quit()
        exit()

    def update_screen(self):
        self.tile_world.draw(self.screen)
        self.player1.draw(self.screen)
        self.player2.draw(self.screen)
        self.ui.update_ui(self.screen)
        pygame.display.flip()
