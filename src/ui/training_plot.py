"""
training_plot.py

Visualizes reinforcement-learning training progress.
"""

import matplotlib.pyplot as plt


class TrainingPlot:

    def __init__(self, report):
        """
        Create a training plotter.

        Parameters:
            report:
                A TrainingReport instance.
        """

        self.report = report

    def plot(self, window=10):
        """
        Display the training progress.

        The figure contains:
            1. Reward per episode
            2. Steps per episode
            3. Success rate per episode

        A moving average is shown for each metric.
        """

        episodes = self.report.episode_numbers()

        rewards = self.report.rewards()
        steps = self.report.steps()
        successes = self.report.successes()

        reward_average = (
            self.report.reward_moving_average(window)
        )

        steps_average = (
            self.report.steps_moving_average(window)
        )

        success_average = (
            self.report.success_rate_moving_average(window)
        )

        # -----------------------------------------
        # Reward plot
        # -----------------------------------------

        plt.figure()

        plt.plot(
            episodes,
            rewards,
            label="Reward"
        )

        plt.plot(
            episodes,
            reward_average,
            label=f"{window}-episode average"
        )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Reward")
        plt.legend()
        plt.grid(True)

        # -----------------------------------------
        # Steps plot
        # -----------------------------------------

        plt.figure()

        plt.plot(
            episodes,
            steps,
            label="Steps"
        )

        plt.plot(
            episodes,
            steps_average,
            label=f"{window}-episode average"
        )

        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Steps per Episode")
        plt.legend()
        plt.grid(True)

        # -----------------------------------------
        # Success rate plot
        # -----------------------------------------

        plt.figure()

        plt.plot(
            episodes,
            success_average,
            label=f"{window}-episode success rate"
        )

        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.title("Training Success Rate")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)

        # -----------------------------------------
        # Display all plots
        # -----------------------------------------

        plt.show()