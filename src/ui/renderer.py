"""
renderer.py

Responsible for drawing the GridWorld.

The renderer NEVER changes the environment.
It only displays it.
"""

import pygame

from src.config import *

class Renderer:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        pygame.display.set_caption(
            "RL Teaching Simulator"
        )

        self.clock = pygame.time.Clock()

    # ---------------------------------------------------------
    # Public Draw Function
    # ---------------------------------------------------------

    def draw(self, env):

        self.screen.fill(WHITE)

        self.draw_grid(env)

        self.draw_goal(env)

        self.draw_obstacles(env)

        self.draw_agent(env)

        pygame.display.flip()

        self.clock.tick(FPS)

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------

    def draw_grid(self, env):

        rows, cols = env.get_grid_size()

        for row in range(rows):

            for col in range(cols):

                rect = pygame.Rect(
                    col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )

                pygame.draw.rect(
                    self.screen,
                    LIGHT_GREY,
                    rect,
                    1
                )

    # ---------------------------------------------------------
    # Goal
    # ---------------------------------------------------------

    def draw_goal(self, env):

        row, col = env.get_goal_position()

        rect = pygame.Rect(
            col * CELL_SIZE,
            row * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )

        pygame.draw.rect(
            self.screen,
            GREEN,
            rect
        )

    # ---------------------------------------------------------
    # Obstacles
    # ---------------------------------------------------------

    def draw_obstacles(self, env):

        for row, col in env.get_obstacles():

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(
                self.screen,
                DARK_GREY,
                rect
            )

    # ---------------------------------------------------------
    # Agent
    # ---------------------------------------------------------

    def draw_agent(self, env):

        row, col = env.get_agent_position()

        center_x = col * CELL_SIZE + CELL_SIZE // 2
        center_y = row * CELL_SIZE + CELL_SIZE // 2

        pygame.draw.circle(
            self.screen,
            BLUE,
            (center_x, center_y),
            CELL_SIZE // 3
        )