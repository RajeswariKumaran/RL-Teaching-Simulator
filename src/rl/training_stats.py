"""
training_stats.py

Stores statistics collected during RL training.
"""


class TrainingStats:

    def __init__(self):
        """
        Create an empty training statistics object.
        """

        self.episode_rewards = []
        self.episode_steps = []

        self.exploration_counts = []
        self.exploitation_counts = []

    def record_episode(
        self,
        reward,
        steps,
        exploration_count,
        exploitation_count
    ):
        """
        Record statistics for one completed episode.
        """

        self.episode_rewards.append(reward)

        self.episode_steps.append(steps)

        self.exploration_counts.append(
            exploration_count
        )

        self.exploitation_counts.append(
            exploitation_count
        )

    @property
    def episodes(self):
        """
        Return the number of recorded episodes.
        """

        return len(self.episode_rewards)

    @property
    def average_reward(self):
        """
        Return the average reward per episode.

        Returns 0.0 if no episodes have been recorded.
        """

        if not self.episode_rewards:
            return 0.0

        return sum(self.episode_rewards) / self.episodes

    @property
    def average_steps(self):
        """
        Return the average number of steps per episode.

        Returns 0.0 if no episodes have been recorded.
        """

        if not self.episode_steps:
            return 0.0

        return sum(self.episode_steps) / self.episodes

    @property
    def total_exploration(self):
        """
        Return the total number of exploratory decisions.
        """

        return sum(self.exploration_counts)

    @property
    def total_exploitation(self):
        """
        Return the total number of exploitative decisions.
        """

        return sum(self.exploitation_counts)

    @property
    def exploration_percentage(self):
        """
        Return the percentage of decisions that
        were exploratory.

        Returns 0.0 if no decisions have been recorded.
        """

        total_decisions = (
            self.total_exploration
            + self.total_exploitation
        )

        if total_decisions == 0:
            return 0.0

        return (
            self.total_exploration
            / total_decisions
            * 100
        )

    @property
    def exploitation_percentage(self):
        """
        Return the percentage of decisions that
        were exploitative.

        Returns 0.0 if no decisions have been recorded.
        """

        total_decisions = (
            self.total_exploration
            + self.total_exploitation
        )

        if total_decisions == 0:
            return 0.0

        return (
            self.total_exploitation
            / total_decisions
            * 100
        )