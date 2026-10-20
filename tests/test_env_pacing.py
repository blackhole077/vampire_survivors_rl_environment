"""Step pacing: each step holds its action for action_repeat / game_fps seconds.
No game needed: tests/fake_game.py stands in for X11, screen capture and the clock.
"""

import pytest

from fake_game import install_fake_game
from vampire_survivors_env.env import VampireSurvivorsEnv


@pytest.fixture
def screen(monkeypatch):
    return install_fake_game(monkeypatch)


def step_ends(env, clock, n, between_s=0.0):
    """Step n times, spending between_s outside the env before each step; return each step's end time."""
    ends = []
    for _ in range(n):
        clock.t += between_s
        env.step(0)
        ends.append(clock.t)
    return ends


def test_steps_run_at_15_per_second_by_default(screen):
    env = VampireSurvivorsEnv()
    env.reset()
    ends = step_ends(env, screen.clock, 16)
    assert ends[-1] - ends[0] == pytest.approx(15 * 4 / 60)
    assert env.late_steps == 0


def test_period_follows_action_repeat_and_game_fps(screen):
    env = VampireSurvivorsEnv(action_repeat=2, game_fps=30.0)
    env.reset()
    ends = step_ends(env, screen.clock, 3)
    assert ends[2] - ends[1] == pytest.approx(2 / 30)


def test_time_between_steps_counts_toward_the_next(screen):
    """An agent that thinks for 40 ms between steps still gets one step per 66.7 ms."""
    env = VampireSurvivorsEnv()
    env.reset()
    ends = step_ends(env, screen.clock, 11, between_s=0.040)
    assert ends[-1] - ends[0] == pytest.approx(10 * 4 / 60)
    assert env.late_steps == 0


def test_a_step_more_than_a_period_late_resets_the_schedule(screen):
    env = VampireSurvivorsEnv()
    env.reset()
    env.step(0)
    screen.clock.t += 0.5  # e.g. a gradient update between steps
    start = screen.clock.t
    env.step(0)
    assert env.late_steps == 1
    assert screen.clock.t - start == pytest.approx(4 / 60)  # a full step, no burst of catch-up steps


def test_reset_starts_a_new_schedule(screen):
    env = VampireSurvivorsEnv()
    env.reset()
    env.step(0)
    screen.clock.t += 10.0
    env.reset()
    env.step(0)
    assert env.late_steps == 0
