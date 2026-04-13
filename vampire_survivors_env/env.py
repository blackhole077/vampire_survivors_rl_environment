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
    _LEVELUP_CENTER_MEAN_THRESHOLD,
    _LEVELUP_COL_SLICE,
    _LEVELUP_CONSEC_FRAMES,
    _LEVELUP_ROW_SLICE,
    _MENU_EDGE_THRESHOLD,
    _RED_RATIO_THRESHOLD,
    _RESULTS_PANEL_THRESHOLD,
)
from vampire_survivors_env.utils.window_utils import (
    GameWindow,
    capture_game_window,
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
        self._gameplay_started: bool = False
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
        self._dark_frame_count = 0
        self._gameplay_started = False
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
        info = self._get_info()
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
        return capture_game_window(self._sct, self._window)

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

    # BUG: Arcana selection is being treated as "levelup"
    # BUG: Main menu is treated as "gameover"
    # BUG: Darker levels mislabeled as "levelup" or "gameover"
    def _detect_game_phase(self, frame: np.ndarray) -> str:
        """Classify the current screen into one of: playing, levelup, card_selection, gameover, results.

        Cascade (each step short-circuits):
          1. Center brightness > threshold → levelup
          2. Edge density in card region > threshold → card_selection or levelup
          3. Red-channel dominance > threshold → gameover or results
          4. Default → playing

        The _gameplay_started flag distinguishes card_selection (pre-run menu)
        from levelup (in-run upgrade screen); it is set to True on the first
        'playing' frame and reset on reset().

        Temporal consistency: _LEVELUP_CONSEC_FRAMES consecutive menu-phase
        frames are required before committing, to suppress transient false
        positives (explosions, dense enemy clusters with high edge density).
        """
        import cv2 as _cv2

        h, w = frame.shape[:2]
        eps = 1e-6

        center = frame[h // 4 : 3 * h // 4, w // 6 : 5 * w // 6]
        center_mean = float(center.mean())

        # Step 1 — bright overlay → levelup
        if center_mean > _LEVELUP_CENTER_MEAN_THRESHOLD:
            phase = "levelup"
        else:
            # Step 2 — high edge density → menu overlay
            region = frame[_LEVELUP_ROW_SLICE, _LEVELUP_COL_SLICE]
            gray = _cv2.cvtColor(region, _cv2.COLOR_RGB2GRAY)
            edge_density = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())

            if edge_density > _MENU_EDGE_THRESHOLD:
                phase = "card_selection" if not self._gameplay_started else "levelup"
            else:
                # Step 3 — red tint → gameover or results
                r_mean = float(frame[:, :, 0].mean())
                g_mean = float(frame[:, :, 1].mean())
                b_mean = float(frame[:, :, 2].mean())
                r_ratio = min(r_mean / (g_mean + eps), r_mean / (b_mean + eps))

                if r_ratio > _RED_RATIO_THRESHOLD:
                    phase = (
                        "results"
                        if center_mean > _RESULTS_PANEL_THRESHOLD
                        else "gameover"
                    )
                else:
                    phase = "playing"

        # Temporal consistency: suppress transient menu-phase flickers.
        if phase in ("levelup", "card_selection"):
            self._dark_frame_count += 1
        else:
            self._dark_frame_count = 0
            if phase == "playing":
                self._gameplay_started = True

        if (
            phase in ("levelup", "card_selection")
            and self._dark_frame_count < _LEVELUP_CONSEC_FRAMES
        ):
            return "playing"  # not yet committed

        return phase

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
        self._phase = self._detect_game_phase(self._latest_frame)
        frame = self._preprocess_frame(latest_frame)
        self._frame_buf.append(frame)

        # Pad with the oldest available frame if the buffer isn't full yet.
        frames = list(self._frame_buf)
        while len(frames) < self.frame_stack:
            frames.insert(0, frames[0])

        return {"frames": np.stack(frames, axis=0)}  # (C, H, W)

    def _get_info(self) -> Dict:
        """Return a diagnostic dict with hp, level, time, kills, and current game phase."""
        return {"phase": self._phase}

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
