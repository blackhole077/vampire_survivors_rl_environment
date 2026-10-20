"""Pixel episode handling and key release.  No game needed: tests/fake_game.py
stands in for X11 and screen capture.
"""

import pytest

import vampire_survivors_env.env as env_module
from fake_game import install_fake_game
from vampire_survivors_env.env import VampireSurvivorsEnv


@pytest.fixture
def screen(monkeypatch):
    return install_fake_game(monkeypatch)


@pytest.fixture
def sent(monkeypatch):
    """Each send_input_to_game call's (released, pressed) keycode counts."""
    calls = []
    monkeypatch.setattr(env_module, "send_input_to_game", lambda d, rel, press: calls.append((len(rel), len(press))))
    return calls


def misread_as(env, phase):
    env._detect_game_phase = lambda frame: phase


def test_close_releases_held_keys(screen, sent):
    env = VampireSurvivorsEnv()
    env.reset()
    env.step(2)  # up-right: w and d held
    assert env._held_keys == {"w", "d"}
    env.close()
    assert env._held_keys == frozenset() and sent[-1] == (2, 0) and screen.closed


@pytest.mark.parametrize("phase", ["gameover", "results"])
def test_pixel_phase_drops_action_and_ends_by_default(screen, phase):
    env = VampireSurvivorsEnv()
    env.reset()
    misread_as(env, phase)
    env._extract_survival_seconds = lambda frame: 0
    env.step(1)
    _, _, terminated, _, _ = env.step(3)
    assert "d" not in env._held_keys and terminated == (phase == "results")


@pytest.mark.parametrize("phase", ["gameover", "results"])
def test_without_pixel_episodes_a_misread_phase_changes_nothing(screen, phase):
    env = VampireSurvivorsEnv(pixel_episodes=False)
    env.reset()
    misread_as(env, phase)
    env._extract_survival_seconds = lambda frame: pytest.fail("OCR'd a reward")
    env.step(1)
    _, r, terminated, truncated, info = env.step(3)
    assert env._held_keys == {"d"} and (r, terminated, truncated) == (0.0, False, False)
    assert info["phase"] == phase and "survived_seconds" not in info
