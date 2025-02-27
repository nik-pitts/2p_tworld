# tile_type, walkable, sprite_index, effect, collectable

TILE_MAPPING = {
    0: ("FLOOR", True, (0, 0), None, False),  # Normal floor
    1: ("WALL", False, (0, 1), None, False),  # Solid wall
    2: ("CHIP", True, (0, 2), "COLLECT", True),  # Chip, collectable
    3: ("WATER", True, (0, 3), "DROWN", False),  # Water causes drowning
    4: ("FIRE", True, (0, 4), "BURN", False),  # Fire burns the player
    34: ("SOCKET", False, (2, 2), "UNLOCK", False),  # Locked socket, unlockable
    21: ("EXIT", True, (1, 5), "EXIT", False),  # Exit portal
    12: ("ICE", True, (0, 12), "SLIDE", False),  # Ice, player slides
    47: ("HINT", True, (2, 15), "HINT", False),  # Hint tile
    110: ("PLAYER", True, None, None, False),  # Player (handled separately)
    100: ("KEY", True, (6, 4), "BLUE", True),  # Key, unlocks BLUE door
    101: ("KEY", True, (6, 5), "RED", True),  # Key, unlocks RED door
    102: ("KEY", True, (6, 6), "GREEN", True),  # Key, unlocks GREEN door
    103: ("KEY", True, (6, 7), "YELLOW", True),  # Key, unlocks YELLOW door
    22: ("DOOR", False, (1, 7), "RED", False),  # Door, requires key
    23: ("DOOR", False, (1, 6), "BLUE", False),  # Door, requires key
    24: ("DOOR", False, (1, 8), "GREEN", False),  # Door, requires key
    25: ("DOOR", False, (1, 9), "YELLOW", False),  # Door, requires key
    13: ("FORCE_FLOOR", True, (0, 13), "FORCE_DOWN", False),  # Force floor
    18: ("FORCE_FLOOR", True, (1, 2), "FORCE_UP", False),  # Force floor
    19: ("FORCE_FLOOR", True, (1, 3), "FORCE_RIGHT", False),  # Force floor
    20: ("FORCE_FLOOR", True, (1, 4), "FORCE_LEFT", False),  # Force floor
    104: ("BOOT", True, (6, 8), "WATER", True),  # Boot, allows walking on ice
    105: ("BOOT", True, (6, 9), "FIRE", True),  # Boot, protects from fire
    107: ("BOOT", True, (6, 11), "FORCE", True),  # Boot, no force
    64: ("BEETLE_NORTH", False, (4, 0), "BEETLE_UP", False),  # Beetle, moves north
    65: ("BEETLE_WEST", False, (4, 1), "BEETLE_LEFT", False),  # Beetle, moves west
    66: ("BEETLE_SOUTH", False, (4, 2), "BEETLE_DOWN", False),  # Beetle, moves south
    67: ("BEETLE_EAST", False, (4, 3), "BEETLE_RIGHT", False),  # Beetle, moves east
    10: ("MOVABLE_DIRT_BLOCK", False, (0, 10), "PUSH", False),  # Movable dirt block
    11: ("DIRT", True, (0, 11), None, False),
}
