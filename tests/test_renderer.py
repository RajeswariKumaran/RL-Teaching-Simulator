"""
test_renderer.py

Temporary test program for verifying the renderer.

This file is NOT part of the final application.
It simply displays the GridWorld until the user closes the window.
"""

import pygame

from src.environment.gridworld import GridWorld
from src.ui.renderer import Renderer


def main():

    env = GridWorld()

    renderer = Renderer()

    running = True

    while running:

        # Process events
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

        # Draw environment
        renderer.draw(env)

    pygame.quit()


if __name__ == "__main__":
    main()