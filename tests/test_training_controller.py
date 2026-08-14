"""
Tests for TrainingController.
"""

from src.actions import Action
from src.environment.gridworld import GridWorld
from src.rl.algorithms.q_learning import QLearning
from src.rl.policy import EpsilonGreedyPolicy
from src.rl.qtable import QTable
from src.rl.training_controller import TrainingController
from src.rl.training_stats import TrainingStats


def create_controller():

    environment = GridWorld()

    qtable = QTable()

    policy = EpsilonGreedyPolicy(
        qtable=qtable,
        epsilon=0.0
    )

    learner = QLearning(
        qtable=qtable,
        learning_rate=0.1,
        gamma=0.9
    )

    stats = TrainingStats()

    controller = TrainingController(
        environment=environment,
        policy=policy,
        learner=learner,
        stats=stats,
        max_steps=100
    )

    return controller


def test_controller_starts_training():

    controller = create_controller()

    controller.start()

    assert controller.is_training()


def test_controller_step_updates_step_count():

    controller = create_controller()

    controller.start()

    controller.step()

    assert controller.get_current_step() == 1


def test_controller_records_reward():

    controller = create_controller()

    controller.start()

    controller.step()

    assert isinstance(
        controller.get_current_reward(),
        (int, float)
    )


def test_controller_records_last_action():

    controller = create_controller()

    controller.start()

    controller.step()

    assert controller.get_last_action() is not None


def test_controller_stops_after_episode_finishes():

    controller = create_controller()

    controller.start()

    for _ in range(100):

        controller.step()

        if not controller.is_training():
            break

    assert not controller.is_training()


def test_controller_records_episode_statistics():

    controller = create_controller()

    controller.start()

    for _ in range(100):

        controller.step()

        if not controller.is_training():
            break

    assert controller.stats.episodes == 1