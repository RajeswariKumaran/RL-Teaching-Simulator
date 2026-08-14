"""
train_agent.py

Runs Q-learning training on the GridWorld environment
without using the Pygame interface.
"""

from src.environment.gridworld import GridWorld
from src.rl.qtable import QTable
from src.rl.policy import EpsilonGreedyPolicy
from src.rl.algorithms.q_learning import QLearning
from src.rl.training import Trainer
from src.rl.training_report import TrainingReport
from src.ui.training_plot import TrainingPlot


def main():

    # -----------------------------------------
    # Training configuration
    # -----------------------------------------

    episodes = 100
    max_steps = 100

    learning_rate = 0.1
    gamma = 0.9
    epsilon = 0.1

    # -----------------------------------------
    # Create environment
    # -----------------------------------------

    environment = GridWorld()

    # -----------------------------------------
    # Create Q-table
    # -----------------------------------------

    qtable = QTable()

    # -----------------------------------------
    # Create policy
    # -----------------------------------------

    policy = EpsilonGreedyPolicy(
        qtable=qtable,
        epsilon=epsilon
    )

    # -----------------------------------------
    # Create Q-learning algorithm
    # -----------------------------------------

    learner = QLearning(
        qtable=qtable,
        learning_rate=learning_rate,
        gamma=gamma
    )

    # -----------------------------------------
    # Create trainer
    # -----------------------------------------

    trainer = Trainer(
        environment=environment,
        policy=policy,
        learner=learner
    )

    # -----------------------------------------
    # Run training
    # -----------------------------------------

    for episode in range(1, episodes + 1):

        total_reward, steps = trainer.run_episode(
            max_steps=max_steps
        )

        print(
            f"Episode {episode:3d} | "
            f"Reward: {total_reward:4.1f} | "
            f"Steps: {steps:3d}"
        )
    report = TrainingReport(trainer.stats)


    # -----------------------------------------
    # Training summary
    # -----------------------------------------

    stats = trainer.stats

    print()
    print("Training complete")
    print("-----------------")

    print(f"Episodes: {stats.episodes}")

    print(
        f"Successful episodes: "
        f"{stats.successful_episodes}"
    )

    print(
        f"Success rate: "
        f"{stats.success_rate:.2f}%"
    )

    print(
        f"Average reward: "
        f"{stats.average_reward:.2f}"
    )

    print(
        f"Average steps: "
        f"{stats.average_steps:.2f}"
    )

    print(
        f"Exploration: "
        f"{stats.exploration_percentage:.2f}%"
    )

    print(
        f"Exploitation: "
        f"{stats.exploitation_percentage:.2f}%"
    )

    # -----------------------------------------
    # Recent training performance
    # -----------------------------------------

    window = 10

    print()
    print(f"Last {window} episodes")
    print("---------------------")

    print(
        f"Average reward: "
        f"{stats.average_reward_last(window):.2f}"
    )

    print(
        f"Average steps: "
        f"{stats.average_steps_last(window):.2f}"
    )

    print(
        f"Success rate: "
        f"{stats.success_rate_last(window):.2f}%"
    )

    # -----------------------------------------
    # Learning progress
    # -----------------------------------------

    report = TrainingReport(stats)

    window = 10

    print()
    print(f"Learning progress - last {window} episodes")
    print("--------------------------------------------")

    print(
        f"Reward moving average: "
        f"{report.reward_moving_average(window)[-1]:.2f}"
    )

    print(
        f"Steps moving average: "
        f"{report.steps_moving_average(window)[-1]:.2f}"
    )

    print(
        f"Success moving average: "
        f"{report.success_rate_moving_average(window)[-1]:.2f}%"
    )

    plotter = TrainingPlot(report)
    plotter.plot(window=10)
    
if __name__ == "__main__":
    main()