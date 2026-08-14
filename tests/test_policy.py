import random

from src.actions import Action
from src.rl.qtable import QTable
from src.rl.policy import (
    GreedyPolicy,
    EpsilonGreedyPolicy,
    DecisionType
)


def test_greedy_policy_selects_best_action():

    qtable = QTable()

    state = (2, 3)

    qtable.set_q_value(state, Action.UP, 1.0)
    qtable.set_q_value(state, Action.DOWN, 2.0)
    qtable.set_q_value(state, Action.LEFT, 0.5)
    qtable.set_q_value(state, Action.RIGHT, 5.0)

    policy = GreedyPolicy(qtable)

    decision = policy.select_action(state)

    assert decision.action == Action.RIGHT

    assert decision.decision_type == DecisionType.EXPLOITATION

    assert decision.random_value is None


def test_epsilon_greedy_explores_when_random_value_is_low(
    monkeypatch
):

    qtable = QTable()

    state = (2, 3)

    qtable.set_q_value(
        state,
        Action.RIGHT,
        5.0
    )

    policy = EpsilonGreedyPolicy(
        qtable,
        epsilon=0.1
    )

    monkeypatch.setattr(
        random,
        "random",
        lambda: 0.05
    )

    decision = policy.select_action(state)

    assert decision.action in list(Action)

    assert decision.decision_type == DecisionType.EXPLORATION

    assert decision.random_value == 0.05


def test_epsilon_greedy_exploits_when_random_value_is_high(
    monkeypatch
):

    qtable = QTable()

    state = (2, 3)

    qtable.set_q_value(state, Action.UP, 1.0)
    qtable.set_q_value(state, Action.DOWN, 2.0)
    qtable.set_q_value(state, Action.LEFT, 0.5)
    qtable.set_q_value(state, Action.RIGHT, 5.0)

    policy = EpsilonGreedyPolicy(
        qtable,
        epsilon=0.1
    )

    monkeypatch.setattr(
        random,
        "random",
        lambda: 0.5
    )

    decision = policy.select_action(state)

    assert decision.action == Action.RIGHT

    assert decision.decision_type == DecisionType.EXPLOITATION

    assert decision.random_value == 0.5