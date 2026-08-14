from src.environment.gridworld import GridWorld
from src.rl.qtable import QTable
from src.rl.policy import EpsilonGreedyPolicy
from src.rl.algorithms.q_learning import QLearning
from src.rl.training import Trainer


def test_gridworld_training_episode():

    # -----------------------------------------
    # Create GridWorld
    # -----------------------------------------

    environment = GridWorld()

    # -----------------------------------------
    # Create Q-table
    # -----------------------------------------

    qtable = QTable()

    # -----------------------------------------
    # Create epsilon-greedy policy
    # -----------------------------------------

    policy = EpsilonGreedyPolicy(
        qtable=qtable,
        epsilon=0.1
    )

    # -----------------------------------------
    # Create Q-learning algorithm
    # -----------------------------------------

    learner = QLearning(
        qtable=qtable,
        learning_rate=0.1,
        gamma=0.9
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
    # Run one episode
    # -----------------------------------------

    total_reward, steps = trainer.run_episode(
        max_steps=100
    )

    # -----------------------------------------
    # Verify training completed
    # -----------------------------------------

    assert isinstance(total_reward, (int, float))

    assert isinstance(steps, int)

    assert steps > 0

    assert steps <= 100

def test_gridworld_training_updates_qtable():

    environment = GridWorld()

    qtable = QTable()

    policy = EpsilonGreedyPolicy(
        qtable=qtable,
        epsilon=0.2
    )

    learner = QLearning(
        qtable=qtable,
        learning_rate=0.1,
        gamma=0.9
    )

    trainer = Trainer(
        environment=environment,
        policy=policy,
        learner=learner
    )

    # Train for several episodes.
    for _ in range(20):

        trainer.run_episode(
            max_steps=100
        )

    # Check the states that were visited.
    #
    # The trainer should have created Q-table entries
    # for states encountered during training.

    assert len(qtable.table) > 0

    # Check that at least one state has
    # a non-zero learned Q-value.

    learned = False

    for state in qtable.table:

        state_values = qtable.get_state_values(state)

        for value in state_values.values():

            if value != 0.0:
                learned = True
                break

        if learned:
            break

    assert learned