"""
evaluation_controller.py

Controls step-by-step evaluation of a trained Q-learning agent.

Unlike TrainingController, this controller NEVER updates
the Q-table. It only follows the learned policy.
"""

from src.rl.evaluator import Evaluator


class EvaluationController:

    def __init__(
        self,
        environment,
        qtable,
        max_steps=100
    ):
        """
        Create an evaluation controller.

        Parameters:
            environment: RL environment
            qtable: Trained QTable
            max_steps: Maximum number of steps per episode
        """

        self.environment = environment
        self.qtable = qtable
        self.max_steps = max_steps

        self.evaluator = Evaluator(
            environment=environment,
            qtable=qtable
        )

        self.evaluating = False

        self.state = None
        self.current_reward = 0
        self.current_step = 0
        self.success = False

    def start(self):
        """
        Start a new evaluation episode.
        """

        self.state = self.environment.reset()

        self.current_reward = 0
        self.current_step = 0
        self.success = False

        self.evaluating = True

    def step(self):
        """
        Perform one evaluation step.

        Returns:
            True if the evaluation episode has finished.
        """

        if not self.evaluating:
            return True

        # -----------------------------------------
        # Stop if maximum steps reached
        # -----------------------------------------

        if self.current_step >= self.max_steps:

            self.evaluating = False

            return True

        # -----------------------------------------
        # Select best learned action
        # -----------------------------------------

        action = self.qtable.get_best_action(
            self.state
        )

        # -----------------------------------------
        # Execute action
        # -----------------------------------------

        next_state, reward, done = self.environment.step(
            action
        )

        # -----------------------------------------
        # Record results
        # -----------------------------------------

        self.current_reward += reward
        self.current_step += 1

        self.state = next_state

        # -----------------------------------------
        # Check terminal state
        # -----------------------------------------

        if done:

            self.success = True
            self.evaluating = False

            return True

        return False

    def is_evaluating(self):
        """
        Return True when evaluation is running.
        """

        return self.evaluating

    def get_current_reward(self):
        """
        Return reward accumulated during evaluation.
        """

        return self.current_reward

    def get_current_step(self):
        """
        Return number of steps taken.
        """

        return self.current_step

    def was_successful(self):
        """
        Return whether the goal was reached.
        """

        return self.success

    def stop(self):
        """
        Stop the current evaluation.
        """

        self.evaluating = False