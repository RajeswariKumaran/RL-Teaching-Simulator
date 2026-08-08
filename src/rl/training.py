"""
training.py

Coordinates the interaction between the environment,
policy, and Q-learning algorithm.
"""


class Trainer:

    def __init__(self, environment, policy, learner):
        """
        Create a training manager.

        Parameters:
            environment: RL environment
            policy: Policy used to select actions
            learner: Q-learning algorithm
        """

        self.environment = environment
        self.policy = policy
        self.learner = learner

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

        for _ in range(max_steps):

            # -----------------------------------------
            # 1. Select an action
            # -----------------------------------------

            decision = self.policy.select_action(state)

            action = decision.action

            # -----------------------------------------
            # 2. Execute action
            # -----------------------------------------

            next_state, reward, done = self.environment.step(
                action
            )

            # -----------------------------------------
            # 3. Update Q-value
            # -----------------------------------------

            self.learner.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            # -----------------------------------------
            # 4. Update episode information
            # -----------------------------------------

            total_reward += reward
            steps += 1

            # Move to the next state
            state = next_state

            # -----------------------------------------
            # 5. Stop if episode is finished
            # -----------------------------------------

            if done:
                break

        return total_reward, steps