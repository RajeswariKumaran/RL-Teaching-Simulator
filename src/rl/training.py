"""
training.py

Coordinates the interaction between the environment,
policy, and Q-learning algorithm.
"""

from src.rl.policy import DecisionType
from src.rl.training_stats import TrainingStats


class Trainer:

    def __init__(
        self,
        environment,
        policy,
        learner,
        stats=None
    ):
        """
        Create a training manager.

        Parameters:
            environment: RL environment
            policy: Policy used to select actions
            learner: Q-learning algorithm
            stats: Optional TrainingStats object
        """

        self.environment = environment
        self.policy = policy
        self.learner = learner

        if stats is None:
            stats = TrainingStats()

        self.stats = stats

    def run_episode(self, max_steps=100):
        """
        Run one complete training episode.

        Returns:
            total_reward
            steps
        """

        state = self.environment.reset()

        total_reward = 0
        steps = 0

        exploration_count = 0
        exploitation_count = 0

        for _ in range(max_steps):

            # -----------------------------------------
            # 1. Select an action
            # -----------------------------------------

            decision = self.policy.select_action(state)

            action = decision.action

            # -----------------------------------------
            # 2. Record decision type
            # -----------------------------------------

            if decision.decision_type == DecisionType.EXPLORATION:
                exploration_count += 1

            elif decision.decision_type == DecisionType.EXPLOITATION:
                exploitation_count += 1

            # -----------------------------------------
            # 3. Execute action
            # -----------------------------------------

            next_state, reward, done = self.environment.step(
                action
            )

            # -----------------------------------------
            # 4. Update Q-value
            # -----------------------------------------

            self.learner.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            # -----------------------------------------
            # 5. Update episode information
            # -----------------------------------------

            total_reward += reward
            steps += 1

            state = next_state

            # -----------------------------------------
            # 6. Stop if episode is finished
            # -----------------------------------------

            if done:
                break

        # -----------------------------------------
        # 7. Record episode statistics
        # -----------------------------------------

        self.stats.record_episode(
            reward=total_reward,
            steps=steps,
            exploration_count=exploration_count,
            exploitation_count=exploitation_count
        )

        return total_reward, steps