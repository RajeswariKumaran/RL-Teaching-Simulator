from src.rl.evaluator import Evaluator
from src.rl.qtable import QTable
from src.actions import Action


class FakeEnvironment:

    def __init__(self):
        self.states = [
            "start",
            "middle",
            "goal"
        ]

        self.index = 0

    def reset(self):
        self.index = 0
        return self.states[self.index]

    def step(self, action):

        self.index += 1

        if self.index == 1:
            return "middle", -1, False

        if self.index == 2:
            return "goal", 10, True

        return "goal", 0, True


def test_evaluator_runs_episode():

    environment = FakeEnvironment()
    qtable = QTable()

    # Make RIGHT the best action.
    qtable.set_q_value(
        "start",
        Action.RIGHT,
        5.0
    )

    qtable.set_q_value(
        "middle",
        Action.RIGHT,
        5.0
    )

    evaluator = Evaluator(
        environment,
        qtable
    )

    reward, steps, success = evaluator.run_episode()

    assert reward == 9
    assert steps == 2
    assert success is True


def test_evaluator_does_not_update_qtable():

    environment = FakeEnvironment()
    qtable = QTable()

    qtable.set_q_value(
        "start",
        Action.RIGHT,
        5.0
    )

    qtable.set_q_value(
        "middle",
        Action.RIGHT,
        5.0
    )

    evaluator = Evaluator(
        environment,
        qtable
    )

    before_start = qtable.get_q_value(
        "start",
        Action.RIGHT
    )

    before_middle = qtable.get_q_value(
        "middle",
        Action.RIGHT
    )

    evaluator.run_episode()

    after_start = qtable.get_q_value(
        "start",
        Action.RIGHT
    )

    after_middle = qtable.get_q_value(
        "middle",
        Action.RIGHT
    )

    assert before_start == after_start
    assert before_middle == after_middle


def test_evaluator_uses_best_action():

    environment = FakeEnvironment()
    qtable = QTable()

    qtable.set_q_value(
        "start",
        Action.RIGHT,
        10.0
    )

    qtable.set_q_value(
        "start",
        Action.LEFT,
        1.0
    )

    evaluator = Evaluator(
        environment,
        qtable
    )

    reward, steps, success = evaluator.run_episode()

    assert success is True
    assert steps == 2


def test_evaluate_multiple_episodes():

    environment = FakeEnvironment()
    qtable = QTable()

    qtable.set_q_value(
        "start",
        Action.RIGHT,
        5.0
    )

    qtable.set_q_value(
        "middle",
        Action.RIGHT,
        5.0
    )

    evaluator = Evaluator(
        environment,
        qtable
    )

    results = evaluator.evaluate(
        episodes=5
    )

    assert results["episodes"] == 5
    assert results["successful_episodes"] == 5
    assert results["success_rate"] == 100.0
    assert results["average_reward"] == 9.0
    assert results["average_steps"] == 2.0