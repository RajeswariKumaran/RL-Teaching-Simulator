"""
training_controller.py

Controls step-by-step reinforcement-learning training
for the interactive Pygame simulator.

Unlike Trainer.run_episode(), this controller does not
run an entire episode at once.

The Pygame application can call step() once per frame
so the user can watch the agent learn.
"""

from src.rl.policy import DecisionType


class TrainingController:

    def __init__(
        self,
        environment,
        policy,
        learner,
        stats,
        max_steps=100
    ):
        """
        Create a step-by-step training controller.

        Parameters:
            environment: RL environment
            policy: action-selection policy
            learner: Q-learning algorithm
            stats: TrainingStats instance
            max_steps: maximum steps allowed per episode
        """

        self.environment = environment
        self.policy = policy
        self.learner = learner
        self.stats = stats

        self.max_steps = max_steps

        # Training state
        self.training = False
        self.state = None

        # Current episode information
        self.current_episode_reward = 0
        self.current_episode_steps = 0

        self.exploration_count = 0
        self.exploitation_count = 0

        # Information for the UI
        self.last_action = None
        self.last_decision_type = None
        self.last_reward = 0
        self.last_done = False

    # ---------------------------------------------------------
    # Start Training
    # ---------------------------------------------------------

    def start(self):
        """
        Start a new training episode.
        """

        self.state = self.environment.reset()

        self.training = True

        self.current_episode_reward = 0
        self.current_episode_steps = 0

        self.exploration_count = 0
        self.exploitation_count = 0

        self.last_action = None
        self.last_decision_type = None
        self.last_reward = 0
        self.last_done = False

    # ---------------------------------------------------------
    # Stop Training
    # ---------------------------------------------------------

    def stop(self):
        """
        Stop step-by-step training.
        """

        self.training = False

    # ---------------------------------------------------------
    # Training Step
    # ---------------------------------------------------------

    def step(self):
        """
        Execute exactly one reinforcement-learning step.

        Returns:
            True if an episode finished.
            False otherwise.
        """

        if not self.training:
            return False

        # -----------------------------------------------------
        # 1. Select action
        # -----------------------------------------------------

        decision = self.policy.select_action(
            self.state
        )

        action = decision.action

        self.last_action = action
        self.last_decision_type = decision.decision_type

        # -----------------------------------------------------
        # 2. Record exploration/exploitation
        # -----------------------------------------------------

        if decision.decision_type == DecisionType.EXPLORATION:

            self.exploration_count += 1

        elif decision.decision_type == DecisionType.EXPLOITATION:

            self.exploitation_count += 1

        # -----------------------------------------------------
        # 3. Execute action
        # -----------------------------------------------------

        next_state, reward, done = self.environment.step(
            action
        )

        self.last_reward = reward
        self.last_done = done

        # -----------------------------------------------------
        # 4. Update Q-value
        # -----------------------------------------------------

        self.learner.update(
            self.state,
            action,
            reward,
            next_state,
            done
        )

        # -----------------------------------------------------
        # 5. Update current episode information
        # -----------------------------------------------------

        self.current_episode_reward += reward
        self.current_episode_steps += 1

        self.state = next_state

        # -----------------------------------------------------
        # 6. Check episode termination
        # -----------------------------------------------------

        if done:

            self._finish_episode(
                success=True
            )

            return True

        # -----------------------------------------------------
        # 7. Check maximum steps
        # -----------------------------------------------------

        if self.current_episode_steps >= self.max_steps:

            self._finish_episode(
                success=False
            )

            return True

        return False

    # ---------------------------------------------------------
    # Finish Episode
    # ---------------------------------------------------------

    def _finish_episode(self, success):
        """
        Record the completed episode in TrainingStats.
        """

        self.stats.record_episode(
            reward=self.current_episode_reward,
            steps=self.current_episode_steps,
            exploration_count=self.exploration_count,
            exploitation_count=self.exploitation_count,
            success=success
        )

        self.training = False

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    def is_training(self):
        """
        Return True if an episode is currently being trained.
        """

        return self.training

    # ---------------------------------------------------------
    # Current Episode
    # ---------------------------------------------------------

    def get_current_step(self):
        """
        Return the number of steps taken in the current episode.
        """

        return self.current_episode_steps

    def get_current_reward(self):
        """
        Return the accumulated reward for the current episode.
        """

        return self.current_episode_reward

    # ---------------------------------------------------------
    # Last Decision
    # ---------------------------------------------------------

    def get_last_action(self):
        """
        Return the last action selected by the policy.
        """

        return self.last_action

    def get_last_decision_type(self):
        """
        Return the last exploration/exploitation decision.
        """

        return self.last_decision_type

    def get_last_reward(self):
        """
        Return the reward from the last environment step.
        """

        return self.last_reward