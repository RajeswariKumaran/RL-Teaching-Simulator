"""
main.py

Entry point for the GridWorld RL Teaching Simulator.

Responsibilities
----------------
- Create the environment
- Create the renderer
- Process user input
- Execute actions
- Redraw the environment

This file deliberately contains NO reinforcement learning.
"""

import pygame

from src.environment.gridworld import GridWorld
from src.ui.renderer import Renderer
from src.actions import KEY_TO_ACTION


def main():

    # -------------------------------------------------
    # Create Environment
    # -------------------------------------------------

    env = GridWorld()

    # -------------------------------------------------
    # Create Renderer
    # -------------------------------------------------

    renderer = Renderer()

    running = True

    # -------------------------------------------------
    # Main Loop
    # -------------------------------------------------

    while running:

        # ---------------------------------------------
        # Process Events
        # ---------------------------------------------

        for event in pygame.event.get():

            # Close window
            if event.type == pygame.QUIT:
                running = False

            # Keyboard input
            elif event.type == pygame.KEYDOWN:

                # Exit
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Reset environment
                elif event.key == pygame.K_r:
                    env.reset()

                # Movement
                elif event.key in KEY_TO_ACTION:

                    if env.is_terminal():
                        print("Episode finished. Press R to start a new episode.")
                        continue

                    action = KEY_TO_ACTION[event.key]

                    next_state, reward, done = env.step(action)

                    print(
                        f"State: {next_state} | "
                        f"Reward: {reward} | "
                        f"Done: {done}"
                    )

        # ---------------------------------------------
        # Draw
        # ---------------------------------------------

        renderer.draw(env)

    pygame.quit()


if __name__ == "__main__":
    main()