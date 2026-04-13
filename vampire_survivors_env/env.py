import time
from collections import deque
from typing import Any, Dict, Optional, SupportsFloat

import cv2
import gymnasium as gym
import mss
import numpy as np
from ewmhlib import EwmhRoot
from Xlib.display import Display

from vampire_survivors_env.constants import (
    _ACTION_KEYS,
    _LEVELUP_BRIGHTNESS_THRESHOLD,
    _LEVELUP_COL_SLICE,
    _LEVELUP_CONSEC_FRAMES,
    _LEVELUP_ROW_SLICE,
)
from vampire_survivors_env.utils.window_utils import (
    GameWindow,
    find_game_window,
    focus_window,
    generate_keycodes,
    send_input_to_game,
    tap_key,
)


class VampireSurvivorsEnv(gym.Env):
    """Gymnasium environment for the Steam game Vampire Survivors.

    The agent controls only player movement; all weapons fire automatically.
    Game state is read from the live game window via screen capture.
    Level-up choices (weapon/powerup selection) are treated as a separate
    action phase, interleaved with the main movement phase.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        frame_stack: int = 4,
        action_repeat: int = 4,
        resolution: tuple[int, int] = (100, 100),
        reward_scale: float = 1.0,
    ) -> None:
        """Set up observation/action spaces and connect to the game window.

        frame_stack consecutive frames are stacked into a single observation.
        action_repeat holds each action for that many frames before returning.
        resolution is the (H, W) to which captured frames are downsampled.
        reward_scale multiplies all reward signals.
        """
        super().__init__()
        self.render_mode = render_mode
        self.resolution = resolution
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale

        self._ewmh_root = EwmhRoot()
        self._display: Optional[Display] = self._ewmh_root.display
        self._window: Optional[GameWindow] = find_game_window(self._ewmh_root)
        self._keycodes = generate_keycodes(self._display)
        self._held_keys: frozenset[str] = frozenset()

        self._sct = mss.mss()
        self._dark_frame_count: int = 0
        self._latest_frame: Optional[np.ndarray] = None

        # Circular buffer of preprocessed (H, W) float32 frames.
        self._frame_buf: deque[np.ndarray] = deque(maxlen=frame_stack)

        h, w = resolution
        self.observation_space = gym.spaces.Dict(
            {
                # (frame_stack, H, W) — channel-first for Conv2d.
                # H×W is the pre-augmentation storage size (e.g. 100×100);
                # CURL random crops to the final training size (e.g. 84×84)
                # are applied in the training loop, not in the env.
                "frames": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(frame_stack, h, w),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(len(_ACTION_KEYS))

    # ------------------------------------------------------------------
    # Core gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Dict, Dict]:
        """Start a fresh run and return the initial (obs, info).

        Navigates through the main menu to begin a new game, waits for the
        first playable frame, and clears accumulated frame stack buffers.
        """
        # self._navigate_to_quick_start()
        focus_window(self._display, self._window.xid)
        obs = self._get_obs()
        info = {}
        return obs, info
        # raise NotImplementedError

    def step(
        self, action: int
    ) -> tuple[Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Send action to the game, advance action_repeat frames, return transition.

        In MOVEMENT phase the action selects a cardinal/diagonal direction or
        stay.  In LEVELUP phase the action selects one of the presented upgrade
        cards.  Returns (obs, reward, terminated, truncated, info).
        """
        self._send_movement_action(action)
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return the latest raw (unprocessed) game frame for human or rgb_array mode."""
        if self._latest_frame is None:
            return None
        if self.render_mode == "human":
            cv2.imshow(
                "Vampire Survivors", cv2.cvtColor(self._latest_frame, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(1)
        return self._latest_frame

    def close(self) -> None:
        """Release the screen-capture handle and any open input devices."""
        self._sct.close()

    # ------------------------------------------------------------------
    # Screen capture & parsing
    # ------------------------------------------------------------------

    def _capture_screen(self) -> np.ndarray:
        """Grab and return the current game-window pixels as an HxWx3 uint8 RGB array."""
        assert self._window is not None
        region = {
            "top": self._window.y,
            "left": self._window.x,
            "width": self._window.width,
            "height": self._window.height,
        }
        # mss returns BGRA; drop alpha and flip to RGB.
        raw = np.array(self._sct.grab(region), dtype=np.uint8)
        return raw[:, :, 2::-1]  # BGRA -> RGB, shape (H, W, 3)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Center-crop to square, resize to self.resolution, convert to grayscale, normalise to [0, 1].

        Returns a (H, W) float32 array.

        The center crop drops the horizontal strips left/right of the player
        (VS keeps the player centered; those strips are the least informative
        part of the screen).  self.resolution is the *pre-augmentation* size
        stored in the replay buffer — the CURL random crop to the final
        training resolution (e.g. 84x84) is applied at training time, not here.
        INTER_AREA is used because we are always downsampling.
        """
        src_h, src_w = frame.shape[:2]
        side = min(src_h, src_w)
        cy, cx = src_h // 2, src_w // 2
        frame = frame[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]

        out_h, out_w = self.resolution
        small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)  # (H, W) uint8
        return gray.astype(np.float32) / 255.0

    def _detect_game_phase(self, frame: np.ndarray) -> str:
        """Classify the current screen as 'playing' or 'levelup'.

        Uses mean brightness of the region where the "Level Up!" header appears.
        The level-up overlay darkens this region significantly (~100) vs normal
        gameplay (~167), giving a large margin around the threshold.

        Temporal consistency: requires _LEVELUP_CONSEC_FRAMES consecutive dark
        frames before committing to 'levelup', preventing false positives from
        transient dark effects (explosions, dense enemy clusters).

        frame: raw HxWx3 uint8 RGB from _capture_screen (game window pixels,
               no window chrome).  Coordinates below are for 640x480; adjust
               _LEVELUP_ROW_SLICE / _LEVELUP_COL_SLICE if resolution changes.
        """
        region_mean = frame[_LEVELUP_ROW_SLICE, _LEVELUP_COL_SLICE].mean()

        if region_mean < _LEVELUP_BRIGHTNESS_THRESHOLD:
            self._dark_frame_count += 1
        else:
            self._dark_frame_count = 0

        if self._dark_frame_count >= _LEVELUP_CONSEC_FRAMES:
            return "levelup"
        return "playing"

    # ------------------------------------------------------------------
    # Observation & info
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        """Assemble and return the current observation dict from the internal state buffers.

        Captures a fresh frame, pushes it into the ring buffer, then stacks
        all buffered frames into a (frame_stack, H, W) float32 array.  When
        the buffer isn't full yet (early in an episode) the oldest slot is
        repeated via np.resize so the shape is always consistent.
        """
        latest_frame = self._capture_screen()
        self._latest_frame = latest_frame.copy()
        frame = self._preprocess_frame(latest_frame)
        self._frame_buf.append(frame)

        # Pad with the oldest available frame if the buffer isn't full yet.
        frames = list(self._frame_buf)
        while len(frames) < self.frame_stack:
            frames.insert(0, frames[0])

        return {"frames": np.stack(frames, axis=0)}  # (C, H, W)

    def _get_info(self) -> Dict:
        """Return a diagnostic dict with hp, level, time, kills, and current game phase."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """Compute the step reward from delta-kills, delta-XP, HP change, and survival bonus."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _send_movement_action(self, action: int) -> None:
        target = _ACTION_KEYS[action]
        to_release = self._held_keys - target
        to_press = target - self._held_keys

        keys_to_release = [self._keycodes[k] for k in to_release]
        keys_to_press = [self._keycodes[k] for k in to_press]
        send_input_to_game(self._display, keys_to_release, keys_to_press)

        self._display.sync()
        self._held_keys = target

    def _send_levelup_action(self, action: int) -> None:
        """Click or press the key corresponding to the chosen level-up card (1-4)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Game window management
    # ------------------------------------------------------------------

    def _navigate_to_new_run(self) -> None:
        """Drive the main-menu UI (character select, stage select) to start a new run."""
        raise NotImplementedError

    def _navigate_to_quick_start(self) -> None:
        """Focus the game window and tap the Quick Start button.

        Assumes the game is already at the main menu with the cursor on Quick Start.
        Uses set_input_focus + sync before XTEST injection so Wine/Proton accepts the events.
        """
        assert self._window is not None
        focus_window(self._display, self._window.xid)

        # Navigate right to Quick Start, then confirm with Space.
        tap_key(self._display, self._keycodes["d"])
        time.sleep(0.2)
        tap_key(self._display, self._keycodes["space"])
        time.sleep(1.0)  # wait for the game run to load


# ------------------------------------------------------------------
# The Gym-only stuff
# ------------------------------------------------------------------


def get_env(
    render_mode: Optional[str] = None,
    frame_stack: int = 4,
    action_repeat: int = 4,
    resolution: tuple[int, int] = (84, 84),
    reward_scale: float = 1.0,
) -> VampireSurvivorsEnv:
    """Construct and return a ready-to-use VampireSurvivorsEnv with the given settings."""
    raise NotImplementedError


def register_env() -> None:
    """Register VampireSurvivorsEnv-v0 with the Gymnasium registry using get_env as entry point."""
    gym.register(
        id="VampireSurvivorsEnv-v0",
        entry_point=get_env,
    )
