"""Test doubles that let VampireSurvivorsEnv run without X11, Proton, or the game.

install_fake_game(monkeypatch) swaps every X11 / mss touchpoint the env uses
for in-memory fakes and returns a FakeScreen whose frames are deterministic,
so observation bytes are reproducible across runs.
"""

from types import SimpleNamespace

import numpy as np

import vampire_survivors_env.env as env_module
from vampire_survivors_env.utils.window_utils import GameWindow

FAKE_PID = 4242
_FRAME_H, _FRAME_W = 480, 640  # constants.py thresholds are calibrated at 640×480


def synthetic_frame(i: int) -> np.ndarray:
    """Return a deterministic HxWx3 uint8 'gameplay-like' frame for step i.

    Smooth background gradient plus a few moving blocks: dark and low-edge
    enough that the pixel phase detector reads it as 'playing'.
    """
    rng = np.random.default_rng(i)
    ys = np.linspace(0, 1, _FRAME_H, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, _FRAME_W, dtype=np.float32)[None, :]
    base = 30 + 20 * ys + 15 * xs
    frame = np.stack([base * 0.6, base, base * 0.8], axis=-1)
    for _ in range(6):
        y, x = rng.integers(0, _FRAME_H - 20), rng.integers(0, _FRAME_W - 20)
        frame[y : y + 20, x : x + 20] = rng.integers(20, 90, size=3)
    return frame.astype(np.uint8)


class FakeScreen:
    """Stands in for mss + capture_game_window; serves synthetic_frame(0), (1), ..."""

    def __init__(self) -> None:
        self.n_grabs = 0
        self.closed = False

    def capture(self, sct, window) -> np.ndarray:
        frame = synthetic_frame(self.n_grabs)
        self.n_grabs += 1
        return frame

    def close(self) -> None:
        self.closed = True


def install_fake_game(monkeypatch) -> FakeScreen:
    """Patch the env module so constructing VampireSurvivorsEnv needs no real game."""
    screen = FakeScreen()
    display = SimpleNamespace(sync=lambda: None)
    window = GameWindow(xid=1, pid=FAKE_PID, x=0, y=0, width=_FRAME_W, height=_FRAME_H)

    monkeypatch.setattr(env_module, "EwmhRoot", lambda: SimpleNamespace(display=display))
    monkeypatch.setattr(env_module, "find_game_window", lambda root: window)
    monkeypatch.setattr(env_module, "generate_keycodes", lambda d: {k: 0 for k in "wasd"} | {"space": 0, "Escape": 0})
    monkeypatch.setattr(env_module.mss, "mss", lambda: screen)
    monkeypatch.setattr(env_module, "capture_game_window", screen.capture)
    monkeypatch.setattr(env_module, "focus_window", lambda d, xid: None)
    monkeypatch.setattr(env_module, "send_input_to_game", lambda d, rel, press: None)
    return screen
