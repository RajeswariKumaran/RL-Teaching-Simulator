from src.rl.evaluation_controller import EvaluationController
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


def create_controller():

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

    controller = EvaluationController(
        environment=environment,
        qtable=qtable
    )

    return controller


def test_evaluation_starts():

    controller = create_controller()

    controller.start()

    assert controller.is_evaluating() is True
    assert controller.get_current_step() == 0
    assert controller.get_current_reward() == 0
    assert controller.was_successful() is False


def test_evaluation_step():

    controller = create_controller()

    controller.start()

    finished = controller.step()

    assert finished is False
    assert controller.is_evaluating() is True
    assert controller.get_current_step() == 1
    assert controller.get_current_reward() == -1


def test_evaluation_reaches_goal():

    controller = create_controller()

    controller.start()

    controller.step()

    finished = controller.step()

    assert finished is True
    assert controller.is_evaluating() is False
    assert controller.get_current_step() == 2
    assert controller.get_current_reward() == 9
    assert controller.was_successful() is True


def test_evaluation_does_not_update_qtable():

    controller = create_controller()

    qtable = controller.qtable

    before_start = qtable.get_q_value(
        "start",
        Action.RIGHT
    )

    before_middle = qtable.get_q_value(
        "middle",
        Action.RIGHT
    )

    controller.start()

    controller.step()
    controller.step()

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


def test_evaluation_uses_best_action():

    controller = create_controller()

    qtable = controller.qtable

    qtable.set_q_value(
        "start",
        Action.LEFT,
        100.0
    )

    qtable.set_q_value(
        "start",
        Action.RIGHT,
        5.0
    )

    controller.start()

    controller.step()

    # LEFT would not move the fake environment toward
    # the expected next state, so the controller should
    # follow the highest Q-value.
    assert controller.get_current_step() == 1


def test_evaluation_stops_at_max_steps():

    controller = create_controller()

    controller.max_steps = 1

    controller.start()

    finished = controller.step()

    assert finished is False
    assert controller.is_evaluating() is True

    finished = controller.step()

    assert finished is True
    assert controller.is_evaluating() is False