"""
actions.py

Defines the action space for the GridWorld environment.

Each action stores:

- integer ID
- display name
- movement vector (row_delta, col_delta)

This allows the environment to move the agent without
large if/elif statements.
"""

from enum import Enum

import pygame


class Action(Enum):
    """
    Enumeration of all valid actions.
    """

    UP = (0, "UP", (-1, 0))
    DOWN = (1, "DOWN", (1, 0))
    LEFT = (2, "LEFT", (0, -1))
    RIGHT = (3, "RIGHT", (0, 1))

    def __init__(self, action_id, display_name, delta):

        self.action_id = action_id
        self.display_name = display_name
        self.delta = delta


# -------------------------------------------------
# Keyboard → Action mapping
# -------------------------------------------------

KEY_TO_ACTION = {
    pygame.K_UP: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
}


# -------------------------------------------------
# Useful list of all actions
# -------------------------------------------------

ALL_ACTIONS = list(Action)