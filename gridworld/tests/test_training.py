from src.actions import Action
from src.rl.qtable import QTable
from src.rl.policy import (
    GreedyPolicy,
    EpsilonGreedyPolicy
)
from src.rl.algorithms.q_learning import QLearning
from src.rl.training import Trainer


class FakeEnvironment:

    def __init__(self):
        self.current_state = (0, 0)
        self.step_count = 0

    def reset(self):
        self.current_state = (0, 0)
        self.step_count = 0
        return self.current_state

    def step(self, action):

        self.step_count += 1

        next_state = (0, self.step_count)

        # Episode finishes after one step.
        reward = 10
        done = True

        return next_state, reward, done


def test_trainer_runs_one_episode():

    # -----------------------------------------
    # Create components
    # -----------------------------------------

    environment = FakeEnvironment()

    qtable = QTable()

    policy = GreedyPolicy(qtable)

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

    # -----------------------------------------
    # Run episode
    # -----------------------------------------

    total_reward, steps = trainer.run_episode()

    # -----------------------------------------
    # Verify episode
    # -----------------------------------------

    assert total_reward == 10

    assert steps == 1

    assert environment.step_count == 1

def test_trainer_updates_qtable():

    environment = FakeEnvironment()

    qtable = QTable()

    policy = GreedyPolicy(qtable)

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

    state = (0, 0)

    # Initially the Q-value should be zero.
    old_q = qtable.get_q_value(
        state,
        Action.UP
    )

    assert old_q == 0.0

    # Run one training episode.
    trainer.run_episode()

    # The policy may choose UP because all
    # initial Q-values are equal.
    new_q = qtable.get_q_value(
        state,
        Action.UP
    )

    assert new_q == 1.0

def test_trainer_records_training_stats():

    environment = FakeEnvironment()

    qtable = QTable()

    policy = GreedyPolicy(qtable)

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

    trainer.run_episode()

    assert trainer.stats.episodes == 1

    assert trainer.stats.episode_rewards == [10]

    assert trainer.stats.episode_steps == [1]

    assert trainer.stats.exploration_counts == [0]

    assert trainer.stats.exploitation_counts == [1]

def test_trainer_records_exploration_and_exploitation(
    monkeypatch
):

    environment = FakeEnvironment()

    qtable = QTable()

    policy = EpsilonGreedyPolicy(
        qtable=qtable,
        epsilon=0.5
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

    # Force exploration.
    monkeypatch.setattr(
        "src.rl.policy.random.random",
        lambda: 0.1
    )

    trainer.run_episode()

    assert trainer.stats.exploration_counts == [1]

    assert trainer.stats.exploitation_counts == [0]