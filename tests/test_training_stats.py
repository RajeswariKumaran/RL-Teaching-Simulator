from src.rl.training_stats import TrainingStats


def test_training_stats_records_episode():

    stats = TrainingStats()

    stats.record_episode(
        reward=10,
        steps=5,
        exploration_count=2,
        exploitation_count=3
    )

    assert stats.episode_rewards == [10]

    assert stats.episode_steps == [5]

    assert stats.exploration_counts == [2]

    assert stats.exploitation_counts == [3]

    assert stats.episodes == 1

def test_training_stats_calculates_averages():

    stats = TrainingStats()

    stats.record_episode(
        reward=10,
        steps=5,
        exploration_count=2,
        exploitation_count=3
    )

    stats.record_episode(
        reward=20,
        steps=10,
        exploration_count=1,
        exploitation_count=9
    )

    assert stats.episodes == 2

    assert stats.average_reward == 15.0

    assert stats.average_steps == 7.5

def test_training_stats_calculates_exploration_percentage():

    stats = TrainingStats()

    stats.record_episode(
        reward=10,
        steps=10,
        exploration_count=2,
        exploitation_count=8
    )

    assert stats.total_exploration == 2

    assert stats.total_exploitation == 8

    assert stats.exploration_percentage == 20.0

    assert stats.exploitation_percentage == 80.0