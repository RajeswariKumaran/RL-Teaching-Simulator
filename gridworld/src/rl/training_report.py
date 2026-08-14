"""
training_report.py

Transforms TrainingStats data into information
that can be used for reporting and visualization.
"""


class TrainingReport:

    def __init__(self, stats):
        """
        Create a training report from TrainingStats.

        Parameters:
            stats:
                A TrainingStats instance containing
                the results of training.
        """

        self.stats = stats

    def episode_numbers(self):
        """
        Return episode numbers starting from 1.
        """

        return list(
            range(1, self.stats.episodes + 1)
        )

    def rewards(self):
        """
        Return the reward history.
        """

        return self.stats.get_reward_history()

    def steps(self):
        """
        Return the steps history.
        """

        return self.stats.get_steps_history()

    def successes(self):
        """
        Return the success history.
        """

        return self.stats.get_success_history()

    def moving_average(self, values, window=10):
        """
        Calculate a moving average.

        For each position, the average is calculated
        using up to the most recent `window` values.

        Example with window=3:

            values = [1, 2, 3, 4]

            result = [
                1,
                1.5,
                2.0,
                3.0
            ]
        """

        if not values:
            return []

        averages = []

        for index in range(len(values)):

            start = max(
                0,
                index - window + 1
            )

            recent_values = values[start:index + 1]

            average = (
                sum(recent_values)
                / len(recent_values)
            )

            averages.append(average)

        return averages

    def reward_moving_average(self, window=10):
        """
        Return the moving average of episode rewards.
        """

        return self.moving_average(
            self.rewards(),
            window
        )

    def steps_moving_average(self, window=10):
        """
        Return the moving average of episode steps.
        """

        return self.moving_average(
            self.steps(),
            window
        )

    def success_rate_moving_average(self, window=10):
        """
        Return the moving success rate as percentages.

        False = 0
        True  = 100
        """

        success_values = [
            100 if success else 0
            for success in self.successes()
        ]

        return self.moving_average(
            success_values,
            window
        )