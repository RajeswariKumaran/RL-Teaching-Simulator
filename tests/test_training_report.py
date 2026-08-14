"""
Tests for TrainingReport.
"""

from src.rl.training_stats import TrainingStats
from src.rl.training_report import TrainingReport


def create_test_stats():
    """
    Create a TrainingStats object with known data.
    """

    stats = TrainingStats()

    stats.record_episode(
        reward=-10,
        steps=20,
        exploration_count=5,
        exploitation_count=15,
        success=False
    )

    stats.record_episode(
        reward=-5,
        steps=15,
        exploration_count=3,
        exploitation_count=12,
        success=False
    )

    stats.record_episode(
        reward=0,
        steps=11,
        exploration_count=2,
        exploitation_count=9,
        success=True
    )

    stats.record_episode(
        reward=3,
        steps=8,
        exploration_count=1,
        exploitation_count=7,
        success=True
    )

    return stats


def test_episode_numbers():

    stats = create_test_stats()

    report = TrainingReport(stats)

    assert report.episode_numbers() == [1, 2, 3, 4]


def test_reward_history():

    stats = create_test_stats()

    report = TrainingReport(stats)

    assert report.rewards() == [-10, -5, 0, 3]


def test_steps_history():

    stats = create_test_stats()

    report = TrainingReport(stats)

    assert report.steps() == [20, 15, 11, 8]


def test_success_history():

    stats = create_test_stats()

    report = TrainingReport(stats)

    assert report.successes() == [
        False,
        False,
        True,
        True
    ]


def test_moving_average():

    stats = create_test_stats()

    report = TrainingReport(stats)

    values = [1, 2, 3, 4]

    result = report.moving_average(
        values,
        window=3
    )

    assert result == [
        1.0,
        1.5,
        2.0,
        3.0
    ]


def test_reward_moving_average():

    stats = create_test_stats()

    report = TrainingReport(stats)

    result = report.reward_moving_average(
        window=2
    )

    assert result == [
        -10.0,
        -7.5,
        -2.5,
        1.5
    ]


def test_steps_moving_average():

    stats = create_test_stats()

    report = TrainingReport(stats)

    result = report.steps_moving_average(
        window=2
    )

    assert result == [
        20.0,
        17.5,
        13.0,
        9.5
    ]


def test_success_rate_moving_average():

    stats = create_test_stats()

    report = TrainingReport(stats)

    result = report.success_rate_moving_average(
        window=2
    )

    assert result == [
        0.0,
        0.0,
        50.0,
        100.0
    ]


def test_moving_average_empty_values():

    stats = create_test_stats()

    report = TrainingReport(stats)

    assert report.moving_average(
        [],
        window=3
    ) == []