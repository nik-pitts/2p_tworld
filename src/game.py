import pygame
import random
import src.settings as settings
from src.player import Player
from src.tiles import TileSpriteSheet, TileWorld
from src.ui import GameUI
from src.agent import (
    RuleBasedAgent,
    TreeBasedAgent,
    BehaviorClonedAgent,
    BehaviorClonedAgentLv2,
    RLAgent,
)
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
        self.background_image = pygame.image.load("./res/backgroundimg_extended.png")

        # Load tile sprite sheet
        self.tile_sprite_sheet = TileSpriteSheet(
            settings.TILE_SHEET_PATH, settings.TILE_SIZE
        )
        self.tile_world = TileWorld(settings.LEVEL_DATA_PATH, self.tile_sprite_sheet)

        # Load game
        self.load_game(next_level=True)

    def load_game(self, next_level=False):
        """Loads a new level or restarts the game."""
        if next_level:
            self.tile_world.level_index += 1
        else:
            self.tile_world.level_index = self.tile_world.level_index

        self.pause = False
        self.level_completed_time = 0
        self.touchdown_time_started = 0
        self.optime_has_triggered = False
        self.op_time_enabled = False
        self.touchdown_time_enabled = False
        self.op_time_start = 0
        self.op_time_duration = 0
        self.tile_world.load_level(self.tile_world.level_index)
        self.agent_thinking_settings = {
            "is_thinking": False,
            "init_time": 0,
            "delay_time": 3000,
        }
        self.level_complete = False

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
            record=True,
        )

        self.player2 = BehaviorClonedAgentLv2(
            player_positions[1][0],
            player_positions[1][1],
            self.tile_world,
            self,
            2,
            "./model/lv2_bc_model_9.4.pth",
            is_train=False,
            alignment=0,
        )

        # self.player1 = Player(
        #     player_positions[0][0],
        #     player_positions[0][1],
        #     self.tile_world,
        #     self,
        #     1,
        #     record=True,
        # )

        # self.player2 = Player(
        #     player_positions[1][0],
        #     player_positions[1][1],
        #     self.tile_world,
        #     self,
        #     2,
        #     record=False,
        # )

        self.agent, self.human = [], None

        if isinstance(
            self.player1,
            (
                RuleBasedAgent,
                TreeBasedAgent,
                BehaviorClonedAgent,
                BehaviorClonedAgentLv2,
                RLAgent,
            ),
        ):
            self.agent.append(self.player1)
        else:
            self.human = self.player1

        if isinstance(
            self.player2,
            (
                RuleBasedAgent,
                TreeBasedAgent,
                BehaviorClonedAgent,
                BehaviorClonedAgentLv2,
                RLAgent,
            ),
        ):
            self.agent.append(self.player2)
        else:
            self.human = self.player2

        # Initialize UI
        self.ui = GameUI(self.tile_world, self.player1, self.player2, self)
        self.tile_world.game_ui = self.ui  # Allow TileWorld to reference UI
        self.recording = (
            f"./data/human_play_data_level{self.tile_world.level_index}.json"
        )

    def check_level_complete(self):
        """if both players have reached the exit."""
        if (
            self.player1.exited or self.player2.exited
        ) and self.touchdown_time_started == 0:
            self.touchdown_time_started = pygame.time.get_ticks()
            self.touchdown_time_enabled = True
        if self.player1.exited and self.player2.exited:
            print(f"🎉 Level {self.tile_world.level_index + 1} Complete!")
            self.pause = True
            self.level_complete = True
            self.level_completed_time = pygame.time.get_ticks()
            self.ui.show_game_result(self.screen)

    def restart_game(self):
        """Restarts the game after game over."""
        print("Restarting game...")
        if self.human:
            save_human_data(self.human, self.recording)
        self.running = True
        self.load_game(next_level=False)

    def check_game_over(self):
        """Ends the game when both players are dead."""
        if not (self.player1.alive or self.player2.alive):
            print("💀💀 Both players are dead! Game over.")
            self.pause = True  # Pause game loop and wait for event
            self.ui.show_game_result(self.screen)

    def run(self):
        """Main game loop."""
        while self.running:
            self.screen.blit(self.background_image, (0, 0))

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    if self.human:
                        save_human_data(self.human, self.recording)
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print(f"Mouse Click at {event.pos}")
                    self.ui.handle_click(event)

                # Block player input when paused or being forced
                if (
                    not self.pause
                    and self.human
                    and not self.op_time_enabled
                    and not self.human.is_being_forced
                ):
                    if event.type == pygame.KEYDOWN:
                        self.human.move(event)

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
                    if self.player1.record:
                        self.player1.log_move(
                            self.player1.direction, None, "death_beetle_collision"
                        )
                    self.check_game_over()

                if self.player2.collision_detection(self.player2.x, self.player2.y):
                    self.player2.remove_self()
                    if self.player2.record:
                        self.player2.log_move(
                            self.player2.direction, None, "death_beetle_collision"
                        )
                    self.check_game_over()

                if self.check_timeout():
                    # Skip the rest of the loop if there was a timeout
                    self.pause = True

                # Player 2's turn (only if it's an agent and no animations are playing)
                if not player1_animated and not player2_animated:
                    for agent in self.agent:
                        agent.step()
                        # pass

                # Move monsters only if game is running and no animations are playing
                if not player1_animated and not player2_animated:
                    for beetle in self.tile_world.beetles:
                        beetle.move()

                # Check for level completion
                self.check_level_complete()

            elif not self.op_time_enabled:
                # Show Replay and Next Level buttons when paused
                self.ui.show_game_result(self.screen)
            else:
                self.ui.show_resume_button(self.screen)

            # Update UI
            self.ui.update_ui(self.screen)
            pygame.display.flip()
            self.clock.tick(settings.FPS)

        if self.human:
            save_human_data(self.human, self.recording)

        pygame.quit()
        exit()

    def update_screen(self):
        self.tile_world.draw(self.screen)
        self.player1.draw(self.screen)
        self.player2.draw(self.screen)
        self.ui.update_ui(self.screen)
        pygame.display.flip()

    def trigger_optime(self):
        self.agent[0].get_optimized_assignments()

    def check_timeout(self):
        """Checks if the level time has expired and ends the game if so."""
        current_time = pygame.time.get_ticks()

        # Calculate elapsed time, accounting for OPTIME pauses
        elapsed_seconds = (
            current_time - self.ui.start_time - self.op_time_duration
        ) / 1000  # Convert to seconds

        if self.touchdown_time_enabled:
            touchdown_seconds = (current_time - self.touchdown_time_started) / 1000
        else:
            touchdown_seconds = 0

        if elapsed_seconds >= self.tile_world.level_time or touchdown_seconds >= 3:
            print(f"⏰ Time's up! Level {self.tile_world.level_index + 1} failed.")
            # Mark players as dead (timeout)
            if self.player1.alive:
                self.player1.alive = False
            if self.player2.alive:
                self.player2.alive = False
            return True

        return False

    def resume_from_optime(self):
        """Calculate and store the duration of the current OPTIME session."""
        if self.op_time_enabled:
            current_time = pygame.time.get_ticks()
            optime_duration = current_time - self.op_time_start
            self.op_time_duration += optime_duration
