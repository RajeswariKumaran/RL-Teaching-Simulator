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
from src.rl.training_stats import TrainingStats

from src.rl.qtable import QTable
from src.rl.policy import EpsilonGreedyPolicy
from src.rl.algorithms.q_learning import QLearning
# from src.rl.training import Trainer

from src.rl.training_controller import TrainingController
# from src.rl.evaluator import Evaluator
from src.rl.evaluation_controller import EvaluationController


def main():

    # -------------------------------------------------
    # Create Environment
    # -------------------------------------------------

    env = GridWorld()

    # -------------------------------------------------
    # Create Renderer
    # -------------------------------------------------

    renderer = Renderer()
    training_stats = TrainingStats()

    qtable = QTable()

    policy = EpsilonGreedyPolicy(
        qtable=qtable,
        epsilon=0.1
    )

    learner = QLearning(
        qtable=qtable,
        learning_rate=0.1,
        gamma=0.9
    )

    training_controller = TrainingController(
        environment=env,
        policy=policy,
        learner=learner,
        stats=training_stats,
        max_steps=100
    )

    evaluation_controller = EvaluationController(
        environment=env,
        qtable=qtable,
        max_steps=100
    )

    # evaluator = Evaluator(
    #     environment=env,
    #     qtable=qtable
    # )

    evaluation_mode = False

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

                    if evaluation_controller.is_evaluating():
                        evaluation_controller.evaluating = False

                    print("Environment reset.")

                # Adding T key to run a training episode
                elif event.key == pygame.K_t:
                    if not training_controller.is_training():
                        training_controller.start()
                        print("Training started.")

                elif event.key == pygame.K_e:
                    if training_controller.is_training():
                        print("Stop training before starting evaluation.")
                        continue

                    if evaluation_controller.is_evaluating():
                        print("Evaluation is already running.")
                        continue

                    evaluation_controller.start()

                    print("Evaluation started.")

                # Movement
                elif event.key in KEY_TO_ACTION:

                    if training_controller.is_training():
                        print("Manual movement disabled during training.")
                        continue

                    if evaluation_controller.is_evaluating():
                        print("Manual movement disabled during evaluation.")
                        continue

                    if env.is_terminal():
                        print(
                            "Episode finished. "
                            "Press R to start a new episode."
                        )
                        continue

                    action = KEY_TO_ACTION[event.key]

                    next_state, reward, done = env.step(action)

                    print(
                        f"State: {next_state} | "
                        f"Reward: {reward} | "
                        f"Done: {done}"
                    )

        if training_controller.is_training():

            episode_finished = training_controller.step()

            if episode_finished:

                print(
                    f"Episode "
                    f"{training_stats.episodes} complete | "
                    f"Reward: "
                    f"{training_controller.get_current_reward()} | "
                    f"Steps: "
                    f"{training_controller.get_current_step()}"
                )

        if evaluation_controller.is_evaluating():

            evaluation_finished = evaluation_controller.step()

            if evaluation_finished:

                print()
                print("Evaluation complete")
                print("-------------------")
                print(
                    f"Success: "
                    f"{evaluation_controller.was_successful()}"
                )
                print(
                    f"Reward: "
                    f"{evaluation_controller.get_current_reward()}"
                )
                print(
                    f"Steps: "
                    f"{evaluation_controller.get_current_step()}"
                )
        # ---------------------------------------------
        # Draw
        # ---------------------------------------------
        
        renderer.draw(
            env,
            training_stats,
            training_controller,
            qtable,
            evaluation_controller
        )

    pygame.quit()


if __name__ == "__main__":
    main()