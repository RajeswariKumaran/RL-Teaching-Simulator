"""
evaluator.py

Evaluates a trained Q-learning agent.

Evaluation does NOT change the Q-table.
The agent always chooses the action with the
highest learned Q-value.
"""


class Evaluator:

    def __init__(self, environment, qtable):
        """
        Create an evaluator.

        Parameters:
            environment: RL environment
            qtable: Trained QTable
        """

        self.environment = environment
        self.qtable = qtable

    def run_episode(self, max_steps=100):
        """
        Run one evaluation episode.

        The Q-table is only read.
        No learning or Q-value updates occur.

        Returns:
            total_reward
            steps
            success
        """

        state = self.environment.reset()

        total_reward = 0
        steps = 0
        success = False

        for _ in range(max_steps):

            # -----------------------------------------
            # 1. Choose the best learned action
            # -----------------------------------------

            action = self.qtable.get_best_action(state)

            # -----------------------------------------
            # 2. Execute action
            # -----------------------------------------

            next_state, reward, done = self.environment.step(
                action
            )

            # -----------------------------------------
            # 3. Record episode information
            # -----------------------------------------

            total_reward += reward
            steps += 1

            state = next_state

            # -----------------------------------------
            # 4. Check for terminal state
            # -----------------------------------------

            if done:
                success = True
                break

        return total_reward, steps, success

    def evaluate(self, episodes=10, max_steps=100):
        """
        Run multiple evaluation episodes.

        Returns:
            dictionary containing evaluation statistics.
        """

        total_rewards = []
        total_steps = []
        successes = 0

        for _ in range(episodes):

            reward, steps, success = self.run_episode(
                max_steps=max_steps
            )

            total_rewards.append(reward)
            total_steps.append(steps)

            if success:
                successes += 1

        average_reward = (
            sum(total_rewards) / episodes
        )

        average_steps = (
            sum(total_steps) / episodes
        )

        success_rate = (
            successes / episodes * 100
        )

        return {
            "episodes": episodes,
            "successful_episodes": successes,
            "success_rate": success_rate,
            "average_reward": average_reward,
            "average_steps": average_steps,
        }