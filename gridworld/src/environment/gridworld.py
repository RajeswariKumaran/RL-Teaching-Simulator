"""
gridworld.py

GridWorld environment.

This class represents the Markov Decision Process (MDP).

Responsibilities
----------------
- Store the current environment state
- Execute actions
- Compute rewards
- Detect terminal states

It deliberately knows NOTHING about:

- Q-learning
- Bellman updates
- Policies
- Neural Networks
"""

from src.actions import Action
from src.config import *


class GridWorld:
    """
    Simple deterministic GridWorld environment.
    """

    def __init__(self):

        self.rows = GRID_ROWS
        self.cols = GRID_COLS

        # Fixed start and goal
        self.start = START_POSITION

        self.goal = GOAL_POSITION

        self.obstacles = OBSTACLES.copy()
        self.reset()

    # ---------------------------------------------------------
    # Reset Environment
    # ---------------------------------------------------------

    def reset(self):

        self.agent = self.start

        self.done = False

        return self.get_state()

    # ---------------------------------------------------------
    # Current State
    # ---------------------------------------------------------

    def get_state(self):

        return self.agent

    # ---------------------------------------------------------
    # Execute Action
    # ---------------------------------------------------------

    def step(self, action: Action):

        if self.done:
            return self.agent, 0, True

        row, col = self.agent

        delta_row, delta_col = action.delta

        candidate_row = row + delta_row
        candidate_col = col + delta_col

        # -----------------------------
        # Check grid boundaries
        # -----------------------------

        if (
            0 <= candidate_row < self.rows
            and
            0 <= candidate_col < self.cols
        ):

            candidate_position = (
                candidate_row,
                candidate_col
            )

            # -----------------------------
            # Check obstacle collision
            # -----------------------------

            if candidate_position not in self.obstacles:

                self.agent = candidate_position

        # -----------------------------
        # Reward
        # -----------------------------

        if self.agent == self.goal:

            self.done = True
            reward = GOAL_REWARD

        else:

            reward = STEP_REWARD

        return self.agent, reward, self.done

    # ---------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Rendering Support
    # ---------------------------------------------------------

    def get_render_data(self):
        """
        Returns all information required by the renderer.

        The renderer should use this method rather than accessing
        internal variables directly.
        """

        return {
            "rows": self.rows,
            "cols": self.cols,
            "agent": self.agent,
            "goal": self.goal,
            "obstacles": self.obstacles,
            "done": self.done,
        }

    def is_terminal(self):

        return self.done

    def is_goal(self, state):

        return state == self.goal

    def is_obstacle(self, state):

        return state in self.obstacles
    
    def is_valid_position(self, position):
        """
        Returns True if the supplied position is inside the grid
        and is not an obstacle.
        """

        row, col = position

        if row < 0 or row >= self.rows:
            return False

        if col < 0 or col >= self.cols:
            return False

        if position in self.obstacles:
            return False

        return True

    # ---------------------------------------------------------
    # Methods used by the renderer
    # ---------------------------------------------------------

    def get_grid_size(self):
        return self.rows, self.cols

    def get_agent_position(self):
        return self.agent

    def get_goal_position(self):
        return self.goal

    def get_obstacles(self):
        return self.obstacles