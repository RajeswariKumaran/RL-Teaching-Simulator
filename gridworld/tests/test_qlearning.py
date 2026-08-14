from src.actions import Action
from src.rl.qtable import QTable
from src.rl.algorithms.q_learning import QLearning
import math 


def test_q_learning_update():

    qtable = QTable()

    learner = QLearning(
        qtable=qtable,
        learning_rate=0.1,
        gamma=0.9
    )

    state = (2, 3)
    next_state = (2, 4)
    action = Action.RIGHT

    reward = -1
    done = False

    # Give the next state a known best Q-value.
    qtable.set_q_value(
        next_state,
        Action.RIGHT,
        2.0
    )

    new_q = learner.update(
        state,
        action,
        reward,
        next_state,
        done
    )

    # Target = -1 + 0.9 * 2
    #        = 0.8
    #
    # New Q = 0 + 0.1 * (0.8 - 0)
    #       = 0.08

    assert math.isclose(new_q, 0.08)

    assert math.isclose(
        qtable.get_q_value(state, action),
        0.08
    )
def test_q_learning_terminal_state():

    qtable = QTable()

    learner = QLearning(
        qtable=qtable,
        learning_rate=0.1,
        gamma=0.9
    )

    state = (3, 4)
    next_state = (4, 4)
    action = Action.RIGHT

    reward = 10
    done = True

    # Even though the next state has a Q-value,
    # it must not contribute because this is terminal.
    qtable.set_q_value(
        next_state,
        Action.RIGHT,
        100.0
    )

    new_q = learner.update(
        state,
        action,
        reward,
        next_state,
        done
    )

    # Target = reward = 10
    #
    # New Q = 0 + 0.1 * (10 - 0)
    #       = 1.0

    assert math.isclose(new_q, 1.0)

    assert math.isclose(
        qtable.get_q_value(state, action),
        1.0
    )