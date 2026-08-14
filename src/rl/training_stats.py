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

        self.episode_successes = []

    def record_episode(
        self,
        reward,
        steps,
        exploration_count,
        exploitation_count,
        success
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

        self.episode_successes.append(
            success
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
        """

        if not self.episode_rewards:
            return 0.0

        return (
            sum(self.episode_rewards)
            / self.episodes
        )

    @property
    def average_steps(self):
        """
        Return the average number of steps per episode.
        """

        if not self.episode_steps:
            return 0.0

        return (
            sum(self.episode_steps)
            / self.episodes
        )

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
        Return the percentage of decisions
        that were exploratory.
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
        Return the percentage of decisions
        that were exploitative.
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

    @property
    def successful_episodes(self):
        """
        Return the number of successful episodes.
        """

        return sum(self.episode_successes)

    @property
    def success_rate(self):
        """
        Return the percentage of successful episodes.
        """

        if self.episodes == 0:
            return 0.0

        return (
            self.successful_episodes
            / self.episodes
            * 100
        )

    def average_reward_last(self, window=10):
        """
        Return the average reward over
        the most recent `window` episodes.
        """

        if not self.episode_rewards:
            return 0.0

        recent_rewards = self.episode_rewards[-window:]

        return (
            sum(recent_rewards)
            / len(recent_rewards)
        )

    def average_steps_last(self, window=10):
        """
        Return the average number of steps over
        the most recent `window` episodes.
        """

        if not self.episode_steps:
            return 0.0

        recent_steps = self.episode_steps[-window:]

        return (
            sum(recent_steps)
            / len(recent_steps)
        )

    def success_rate_last(self, window=10):
        """
        Return the success rate over
        the most recent `window` episodes.
        """

        if not self.episode_successes:
            return 0.0

        recent_successes = self.episode_successes[-window:]

        return (
            sum(recent_successes)
            / len(recent_successes)
            * 100
        )

    def get_reward_history(self):
        """
        Return the reward received in each episode.
        """

        return self.episode_rewards.copy()

    def get_steps_history(self):
        """
        Return the number of steps taken in each episode.
        """

        return self.episode_steps.copy()

    def get_success_history(self):
        """
        Return whether each episode was successful.

        True  = goal reached
        False = goal not reached
        """

        return self.episode_successes.copy()