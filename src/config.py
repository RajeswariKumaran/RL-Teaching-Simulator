"""
config.py

Global configuration values used throughout the RL Teaching Simulator.

Keeping all constants here avoids hard-coded values scattered throughout
the project.
"""

# -------------------------------------------------
# Grid Configuration
# -------------------------------------------------

GRID_ROWS = 5
GRID_COLS = 5

CELL_SIZE = 100

# -------------------------------------------------
# Window Configuration
# -------------------------------------------------

SIDE_PANEL_WIDTH = 350

WINDOW_WIDTH = GRID_COLS * CELL_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE

FPS = 30

# -------------------------------------------------
# Colours
# -------------------------------------------------

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

LIGHT_GREY = (220, 220, 220)
DARK_GREY = (120, 120, 120)

BLUE = (50, 120, 255)
GREEN = (40, 180, 40)
RED = (220, 60, 60)

YELLOW = (240, 220, 0)

# -------------------------------------------------
# Rewards
# -------------------------------------------------

STEP_REWARD = -1
GOAL_REWARD = 10

# -------------------------------------------------
# Environment Configuration
# -------------------------------------------------

START_POSITION = (0, 0)

GOAL_POSITION = (GRID_ROWS - 1, GRID_COLS - 1)

OBSTACLES = {

    (1, 1),
    (1, 2),
    (2, 3),
    (3, 1),

}
# -------------------------------------------------
# Learning Parameters
# (Used in later phases)
# -------------------------------------------------

LEARNING_RATE = 0.10

DISCOUNT_FACTOR = 0.95

EPSILON = 0.10

# -------------------------------------------------
# Font Sizes
# -------------------------------------------------

TITLE_FONT_SIZE = 28
TEXT_FONT_SIZE = 22
SMALL_FONT_SIZE = 16