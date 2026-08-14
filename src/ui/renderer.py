"""
renderer.py

Responsible for drawing the GridWorld.

The renderer NEVER changes the environment.
It only displays it.
"""

import pygame


from src.config import *


class Renderer:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        pygame.display.set_caption(
            "RL Teaching Simulator"
        )

        self.clock = pygame.time.Clock()

        # Font used for training information
        self.font = pygame.font.Font(None, 28)

    # ---------------------------------------------------------
    # Public Draw Function
    # ---------------------------------------------------------

    def draw(
        self,
        env,
        training_stats=None,
        training_controller=None,
        qtable=None,
        evaluation_controller=None
    ):

        self.screen.fill(WHITE)

        self.draw_grid(env)

        self.draw_goal(env)

        self.draw_obstacles(env)

        if qtable is not None:
            self.draw_q_values(env, qtable)

        self.draw_agent(env)


        if training_stats is not None:
            self.draw_training_stats(
                env,
                training_stats
            )

        if training_controller is not None:
            self.draw_current_decision(
                env,
                training_controller
            )

        pygame.display.flip()

        self.clock.tick(FPS)
    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------

    def draw_grid(self, env):

        rows, cols = env.get_grid_size()

        for row in range(rows):

            for col in range(cols):

                rect = pygame.Rect(
                    col * CELL_SIZE,
                    row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )

                pygame.draw.rect(
                    self.screen,
                    LIGHT_GREY,
                    rect,
                    1
                )

    # ---------------------------------------------------------
    # Goal
    # ---------------------------------------------------------

    def draw_goal(self, env):

        row, col = env.get_goal_position()

        rect = pygame.Rect(
            col * CELL_SIZE,
            row * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )

        pygame.draw.rect(
            self.screen,
            GREEN,
            rect
        )

    # ---------------------------------------------------------
    # Obstacles
    # ---------------------------------------------------------

    def draw_obstacles(self, env):

        for row, col in env.get_obstacles():

            rect = pygame.Rect(
                col * CELL_SIZE,
                row * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(
                self.screen,
                DARK_GREY,
                rect
            )

    # ---------------------------------------------------------
    # Agent
    # ---------------------------------------------------------

    def draw_agent(self, env):

        row, col = env.get_agent_position()

        center_x = col * CELL_SIZE + CELL_SIZE // 2
        center_y = row * CELL_SIZE + CELL_SIZE // 2

        pygame.draw.circle(
            self.screen,
            BLUE,
            (center_x, center_y),
            CELL_SIZE // 3
        )

    # ---------------------------------------------------------
    # Training Statistics
    # ---------------------------------------------------------

    def draw_training_stats(self, env, stats):

        # Start the statistics panel to the right
        # of the GridWorld.

        x = GRID_COLS * CELL_SIZE + 20
        y = 20

        title = self.font.render(
            "TRAINING",
            True,
            BLACK
        )

        self.screen.blit(
            title,
            (x, y)
        )

        y += 35

        lines = [
            f"Episodes: {stats.episodes}",
            f"Success: {stats.success_rate:.1f}%",
            f"Avg reward: {stats.average_reward:.2f}",
            f"Avg steps: {stats.average_steps:.2f}",
        ]

        for line in lines:

            text = self.font.render(
                line,
                True,
                BLACK
            )

            self.screen.blit(
                text,
                (x, y)
            )

            y += 28

        # -----------------------------------------------------
        # Recent performance
        # -----------------------------------------------------

        y += 10

        title = self.font.render(
            "LAST 10",
            True,
            BLACK
        )

        self.screen.blit(
            title,
            (x, y)
        )

        y += 35

        recent_lines = [
            f"Success: {stats.success_rate_last(10):.1f}%",
            f"Reward: {stats.average_reward_last(10):.2f}",
            f"Steps: {stats.average_steps_last(10):.2f}",
        ]

        for line in recent_lines:

            text = self.font.render(
                line,
                True,
                BLACK
            )

            self.screen.blit(
                text,
                (x, y)
            )

            y += 28

    # ---------------------------------------------------------
    # Current RL Decision
    # ---------------------------------------------------------

    def draw_current_decision(self, env, controller):

        x = GRID_COLS * CELL_SIZE + 20
        y = 300

        title = self.font.render(
            "CURRENT STEP",
            True,
            BLACK
        )

        self.screen.blit(
            title,
            (x, y)
        )

        y += 35

        action = controller.get_last_action()
        decision = controller.get_last_decision_type()
        reward = controller.get_last_reward()
        step = controller.get_current_step()

        # Convert enum values into readable text
        if action is None:
            action_text = "NONE"
        else:
            action_text = action.name

        if decision is None:
            decision_text = "NONE"
        else:
            decision_text = decision.name

        lines = [
            f"Step: {step}",
            f"Action: {action_text}",
            f"Reward: {reward}",
            f"Decision: {decision_text}",
        ]

        for line in lines:

            text = self.font.render(
                line,
                True,
                BLACK
            )

            self.screen.blit(
                text,
                (x, y)
            )

            y += 28

    # ---------------------------------------------------------
    # Panel Position
    # ---------------------------------------------------------

    def get_panel_x(self, env):

        rows, cols = env.get_grid_size()

        grid_width = cols * CELL_SIZE

        return grid_width + 20

    # ---------------------------------------------------------
    # Q-Value Visualization
    # ---------------------------------------------------------

    def draw_q_values(self, env, qtable):

        for row in range(GRID_ROWS):

            for col in range(GRID_COLS):

                state = (row, col)

                # Don't draw arrows on obstacles
                if state in env.get_obstacles():
                    continue

                # Don't draw an arrow on the goal
                if state == env.get_goal_position():
                    continue

                action = qtable.get_best_action(state)

                self.draw_action_arrow(
                    row,
                    col,
                    action
                )

    # ---------------------------------------------------------
    # Q-Value Action Arrow
    # ---------------------------------------------------------

    def draw_action_arrow(
        self,
        row,
        col,
        action
    ):

        center_x = (
            col * CELL_SIZE
            + CELL_SIZE // 2
        )

        center_y = (
            row * CELL_SIZE
            + CELL_SIZE // 2
        )

        arrow_length = CELL_SIZE // 3

        # Start point
        start_x = center_x
        start_y = center_y

        # End point
        end_x = center_x
        end_y = center_y

        if action.name == "UP":

            end_y -= arrow_length

        elif action.name == "DOWN":

            end_y += arrow_length

        elif action.name == "LEFT":

            end_x -= arrow_length

        elif action.name == "RIGHT":

            end_x += arrow_length

        # Draw the main arrow line
        pygame.draw.line(
            self.screen,
            RED,
            (start_x, start_y),
            (end_x, end_y),
            4
        )

        # Arrow head
        head_size = 10

        if action.name == "UP":

            points = [
                (end_x, end_y),
                (end_x - head_size, end_y + head_size),
                (end_x + head_size, end_y + head_size)
            ]

        elif action.name == "DOWN":

            points = [
                (end_x, end_y),
                (end_x - head_size, end_y - head_size),
                (end_x + head_size, end_y - head_size)
            ]

        elif action.name == "LEFT":

            points = [
                (end_x, end_y),
                (end_x + head_size, end_y - head_size),
                (end_x + head_size, end_y + head_size)
            ]

        else:  # RIGHT

            points = [
                (end_x, end_y),
                (end_x - head_size, end_y - head_size),
                (end_x - head_size, end_y + head_size)
            ]

        pygame.draw.polygon(
            self.screen,
            RED,
            points
        )