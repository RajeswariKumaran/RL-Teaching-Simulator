from src.actions import Action
from src.rl.qtable import QTable


def test_qtable_initializes_values_to_zero():

    qtable = QTable()

    state = (2, 3)

    assert qtable.get_q_value(state, Action.UP) == 0.0
    assert qtable.get_q_value(state, Action.DOWN) == 0.0
    assert qtable.get_q_value(state, Action.LEFT) == 0.0
    assert qtable.get_q_value(state, Action.RIGHT) == 0.0


def test_qtable_can_set_and_get_value():

    qtable = QTable()

    state = (2, 3)

    qtable.set_q_value(
        state,
        Action.RIGHT,
        5.2
    )

    assert qtable.get_q_value(
        state,
        Action.RIGHT
    ) == 5.2


def test_qtable_finds_best_action():

    qtable = QTable()

    state = (2, 3)

    qtable.set_q_value(state, Action.UP, -1.0)
    qtable.set_q_value(state, Action.DOWN, 2.0)
    qtable.set_q_value(state, Action.LEFT, 0.5)
    qtable.set_q_value(state, Action.RIGHT, 5.0)

    assert qtable.get_best_action(state) == Action.RIGHT


def test_qtable_finds_max_q_value():

    qtable = QTable()

    state = (2, 3)

    qtable.set_q_value(state, Action.UP, -1.0)
    qtable.set_q_value(state, Action.DOWN, 2.0)
    qtable.set_q_value(state, Action.LEFT, 0.5)
    qtable.set_q_value(state, Action.RIGHT, 5.0)

    assert qtable.get_max_q(state) == 5.0