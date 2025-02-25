TILE_MAPPING = {
    0: ("FLOOR", True, (0, 0), None),  # Normal floor
    1: ("WALL", False, (0, 1), None),  # Solid wall
    2: ("CHIP", True, (0, 2), "COLLECT"),  # Chip, collectable
    3: ("WATER", True, (0, 3), "DROWN"),  # Water causes drowning
    4: ("FIRE", True, (0, 4), "BURN"),  # Fire burns the player
    34: ("SOCKET", False, (2, 2), "UNLOCK"),  # Locked socket, unlockable
    21: ("EXIT", True, (1, 5), "EXIT"),  # Exit portal
    12: ("ICE", True, (0, 12), "SLIDE"),  # Ice, player slides
    47: ("HINT", True, (2, 15), "HINT"),  # Hint tile
    110: ("PLAYER", True, None, None),  # Player (handled separately)
    100: ("BLUE_KEY", True, (6, 4), "UNLOCK"),  # Key, unlocks BLUE door
    101: ("RED_KEY", True, (6, 5), "UNLOCK"),  # Key, unlocks RED door
    22: ("RED_DOOR", False, (1, 7), None),  # Door, requires key
    23: ("BLUE_DOOR", False, (1, 6), None),  # Door, requires key
    104: ("WATER_BOOT", True, (6, 8), "PROTECT"),  # Boot, allows walking on ice
    105: ("FIRE_BOOT", True, (6, 9), "PROTECT"),  # Boot, protects from fire
}
