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
    100: ("BLUE_KEY", True, (6, 4), "UNLOCK", True),  # Key, unlocks BLUE door
    101: ("RED_KEY", True, (6, 5), "UNLOCK", True),  # Key, unlocks RED door
    102: ("GREEN_KEY", True, (6, 6), "UNLOCK", True),  # Key, unlocks GREEN door
    103: ("YELLOW_KEY", True, (6, 7), "UNLOCK", True),  # Key, unlocks YELLOW door
    22: ("RED_DOOR", False, (1, 7), None, False),  # Door, requires key
    23: ("BLUE_DOOR", False, (1, 6), None, False),  # Door, requires key
    24: ("GREEN_DOOR", False, (1, 8), None, False),  # Door, requires key
    25: ("YELLOW_DOOR", False, (1, 9), None, False),  # Door, requires key
    13: ("FORCE_FLOOR_SOUTH", True, (0, 13), "FORCE", False),  # Force floor
    104: ("WATER_BOOT", True, (6, 8), "PROTECT", True),  # Boot, allows walking on ice
    105: ("FIRE_BOOT", True, (6, 9), "PROTECT", True),  # Boot, protects from fire
    64: ("BEETLE_NORTH", True, (4, 0), "BEETLE", False),  # Beetle, moves north
    65: ("BEETLE_WEST", True, (4, 1), "BEETLE", False),  # Beetle, moves west
    66: ("BEETLE_SOUTH", True, (4, 2), "BEETLE", False),  # Beetle, moves south
    67: ("BEETLE_EAST", True, (4, 3), "BEETLE", False),  # Beetle, moves east
    10: ("MOVABLE_DIRT_BLOCK", False, (0, 10), "PUSH", False),  # Movable dirt block
}
