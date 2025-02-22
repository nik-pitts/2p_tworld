import pygame

def get_tile(sheet, x, y, width, height):
    """Extracts a tile from the sprite sheet."""
    tile = pygame.Surface((width, height), pygame.SRCALPHA)
    tile.blit(sheet, (0, 0), (x*width, y*height, width, height))
    return tile

pygame.init()

TILE_SIZE = 48
MAP_SIZE = 12
UI_SIZE = 5
WIDTH, HEIGHT = TILE_SIZE*(MAP_SIZE+UI_SIZE), TILE_SIZE*MAP_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
tile_sheet = pygame.image.load("../res/tiles.bmp").convert_alpha()

floor_tile = get_tile(tile_sheet, 6, 12, TILE_SIZE, TILE_SIZE)

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen

    # Draw tiles in a grid
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            screen.blit(floor_tile, (col * TILE_SIZE, row * TILE_SIZE))

    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()