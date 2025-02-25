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
        self.tile_sprite_sheet = TileSpriteSheet(
            settings.TILE_SHEET_PATH, settings.TILE_SIZE
        )
        self.tile_world = TileWorld(settings.LEVEL_DATA_PATH, self.tile_sprite_sheet)
        self.load_game(next_level=False)

    def load_game(self, next_level=False):
        if next_level:
            self.tile_world.level_index += 1
            self.tile_world.load_level(self.tile_world.level_index)
        else:
            self.tile_world.level_index = 0
            self.tile_world.load_level(0)

        # Load player positions
        player_positions = self.tile_world.player_positions
        if len(player_positions) < 2:
            raise ValueError(
                "At least two player positions must be defined in the level data!"
            )

        self.player1 = Player(
            player_positions[0][0],
            player_positions[0][1],
            self.tile_world,
            1,
            record=False,
        )

        self.player2 = Player(
            player_positions[1][0],
            player_positions[1][1],
            self.tile_world,
            2,
            record=True,
        )

        # self.player2 = RuleBasedAgent(
        #     player_positions[0][0], player_positions[0][1], self.tile_world, 2
        # )

        # self.player2 = BehaviorClonedAgent(
        #     player_positions[1][0],
        #     player_positions[1][1],
        #     self.tile_world,
        #     2,
        #     "./model/lv1_bc_model_6.0.pth",
        # )

        # UI
        self.ui = GameUI(self.tile_world, self.player1, self.player2, self)

        # Tileworld UI reference
        self.tile_world.game_ui = self.ui

    def check_level_complete(self):
        if self.player1.exited and self.player2.exited:
            print(f"Level {self.tile_world.level_index + 1} Complete!")
            pygame.time.delay(1000)  # Small delay before transition
            self.load_game(next_level=True)

    def restart_game(self):
        print("Restarting game...")
        save_human_data(self.player2)
        self.load_game()

    def run(self):
        """Mian game loop"""
        while self.running:
            self.screen.fill((0, 0, 0))  # 화면 초기화

            # 🔹 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    save_human_data(self.player2)
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    # self.player1.move(event)
                    self.player2.move(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print(f"Mouse xClick at {event.pos}")
                    self.ui.handle_click(event)

            # when P2 is an agent uncomment the following lines
            # if not self.player2.exited:
            #     self.player2.step()

            # Tilemap + player
            self.tile_world.draw(self.screen)
            self.player1.draw(self.screen)
            self.player2.draw(self.screen)

            # UI update
            self.ui.update_ui(self.screen)

            # Check level completion
            self.check_level_complete()

            pygame.display.flip()
            self.clock.tick(settings.FPS)

        save_human_data(self.player1)
        pygame.quit()
        exit()
