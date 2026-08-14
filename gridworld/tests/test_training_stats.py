from src.rl.training_stats import TrainingStats


def test_training_stats_records_episode():

    stats = TrainingStats()

    stats.record_episode(
        reward=10,
        steps=5,
        exploration_count=2,
        exploitation_count=3,
        success=True
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
        exploitation_count=3,
        success=True
    )

    stats.record_episode(
        reward=20,
        steps=10,
        exploration_count=1,
        exploitation_count=9,
        success=True
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
        exploitation_count=8,
        success=True
    )

    assert stats.total_exploration == 2

    assert stats.total_exploitation == 8

    assert stats.exploration_percentage == 20.0

    assert stats.exploitation_percentage == 80.0

def test_training_stats_calculates_success_rate():

    stats = TrainingStats()

    stats.record_episode(
        reward=10,
        steps=5,
        exploration_count=2,
        exploitation_count=3,
        success=True
    )

    stats.record_episode(
        reward=-10,
        steps=20,
        exploration_count=4,
        exploitation_count=16,
        success=False
    )

    stats.record_episode(
        reward=5,
        steps=8,
        exploration_count=1,
        exploitation_count=7,
        success=True
    )

    assert stats.successful_episodes == 2

    assert stats.success_rate == 2 / 3 * 100

def test_training_stats_recent_window():

    stats = TrainingStats()

    stats.record_episode(
        reward=-10,
        steps=20,
        exploration_count=5,
        exploitation_count=15,
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

    stats.record_episode(
        reward=3,
        steps=8,
        exploration_count=1,
        exploitation_count=7,
        success=True
    )

    assert stats.average_reward_last(2) == 3.0

    assert stats.average_steps_last(2) == 8.0

    assert stats.success_rate_last(2) == 100.0

def test_training_stats_window_larger_than_data():

    stats = TrainingStats()

    stats.record_episode(
        reward=10,
        steps=5,
        exploration_count=1,
        exploitation_count=4,
        success=True
    )

    stats.record_episode(
        reward=0,
        steps=10,
        exploration_count=2,
        exploitation_count=8,
        success=False
    )

    assert stats.average_reward_last(10) == 5.0

    assert stats.average_steps_last(10) == 7.5

    assert stats.success_rate_last(10) == 50.0

def test_training_stats_returns_reward_history():

    stats = TrainingStats()

    stats.record_episode(
        reward=-10,
        steps=20,
        exploration_count=5,
        exploitation_count=15,
        success=False
    )

    stats.record_episode(
        reward=3,
        steps=8,
        exploration_count=1,
        exploitation_count=7,
        success=True
    )

    assert stats.get_reward_history() == [-10, 3]


def test_training_stats_returns_steps_history():

    stats = TrainingStats()

    stats.record_episode(
        reward=-10,
        steps=20,
        exploration_count=5,
        exploitation_count=15,
        success=False
    )

    stats.record_episode(
        reward=3,
        steps=8,
        exploration_count=1,
        exploitation_count=7,
        success=True
    )

    assert stats.get_steps_history() == [20, 8]


def test_training_stats_returns_success_history():

    stats = TrainingStats()

    stats.record_episode(
        reward=-10,
        steps=20,
        exploration_count=5,
        exploitation_count=15,
        success=False
    )

    stats.record_episode(
        reward=3,
        steps=8,
        exploration_count=1,
        exploitation_count=7,
        success=True
    )

    assert stats.get_success_history() == [False, True]