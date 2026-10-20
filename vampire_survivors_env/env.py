import re
import time
from collections import deque
from typing import Any, Dict, Optional, SupportsFloat, Union

import cv2
import gymnasium as gym
import mss
import numpy as np
import pytesseract
from ewmhlib import EwmhRoot
from Xlib.display import Display

from vampire_survivors_env.constants import (
    _ACTION_KEYS,
    _CARD_RED_RATIO_THRESHOLD,
    _LEVELUP_CENTER_MEAN_THRESHOLD,
    _LEVELUP_COL_SLICE,
    _LEVELUP_CONSEC_FRAMES,
    _LEVELUP_ROW_SLICE,
    _MENU_CENTER_THRESHOLD,
    _MENU_EDGE_THRESHOLD,
    _RED_RATIO_THRESHOLD,
    _RESULTS_PANEL_THRESHOLD,
    _SURVIVED_OCR_SCALE,
    _SURVIVED_THRESH,
    _SURVIVED_TIME_ROI,
)
from vampire_survivors_env.features.loader import load_provider
from vampire_survivors_env.features.spec import check_compatible, null_frame
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
        game_fps: float = 60.0,
        feature_provider: Union[None, str, object] = None,
        feature_provider_kwargs: Optional[dict[str, Any]] = None,
        pixel_episodes: bool = True,
    ) -> None:
        """Set up observation/action spaces and connect to the game window.

        frame_stack consecutive frames are stacked into a single observation.
        action_repeat holds each action for that many game frames before
        returning: steps are paced to action_repeat / game_fps seconds (15 per
        second at the defaults).  The game runs in real time and can't be
        stepped, so pacing is by wall clock on a fixed schedule; time the
        caller spends between steps counts toward the next one.  A step that
        starts more than a whole step late resets the schedule and is counted
        in self.late_steps.
        resolution is the (H, W) to which captured frames are downsampled.
        reward_scale multiplies all reward signals.
        feature_provider optionally adds structured game-state features (see
        vampire_survivors_env/features/spec.py): a provider instance, a
        "package.module:Name" import path, or an entry-point name, constructed
        with feature_provider_kwargs.  When set, observations gain a "features"
        key and info gains features_ok / features_read_ms / features_error.
        When None (the default) observations and info are pixel-only, as before.
        pixel_episodes=False is for a wrapper that ends episodes itself (e.g.
        from game memory): step() then always sends the action and never
        terminates or OCRs a reward from the pixel phase, which can misfire;
        the phase is still reported in info["phase"].
        """
        super().__init__()
        self.render_mode = render_mode
        self.resolution = resolution
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
        self.game_fps = game_fps
        self.pixel_episodes = pixel_episodes
        self.late_steps: int = 0
        self._step_due: Optional[float] = None  # monotonic time the current step ends

        self._ewmh_root = EwmhRoot()
        self._display: Optional[Display] = self._ewmh_root.display
        self._window: Optional[GameWindow] = find_game_window(self._ewmh_root)
        self._keycodes = generate_keycodes(self._display)
        self._held_keys: frozenset[str] = frozenset()

        self._sct = mss.mss()
        self._dark_frame_count: int = 0
        self._gameplay_started: bool = False
        self._latest_frame: Optional[np.ndarray] = None
        self._terminal_reward: Optional[float] = None
        self._phase: str = "playing"

        # Circular buffer of preprocessed (H, W) float32 frames.
        self._frame_buf: deque[np.ndarray] = deque(maxlen=frame_stack)

        # Optional feature provider.  Config errors (bad import path,
        # incompatible spec version) raise here; runtime errors never do.
        self._provider: Optional[object] = None
        self._feature_space: Optional[gym.spaces.Dict] = None
        self._provider_attached: bool = False
        self._provider_attach_error: Optional[str] = None
        self._feature_status: dict[str, Any] = {}
        if feature_provider is not None:
            self._provider = load_provider(feature_provider, feature_provider_kwargs)
            self._feature_space = check_compatible(self._provider)
            self._attach_provider()

        h, w = resolution
        obs_spaces: dict[str, gym.spaces.Space] = {
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
        if self._provider is not None:
            obs_spaces["features"] = self._feature_space
        self.observation_space = gym.spaces.Dict(obs_spaces)
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
        self._terminal_reward = None
        self._phase = "playing"
        self._step_due = None
        if self._provider is not None and not self._provider_attached:
            self._attach_provider()
        focus_window(self._display, self._window.xid)
        obs = self._get_obs()
        info = {}
        if self._provider is not None:
            info.update(self._feature_status)
        return obs, info
        # raise NotImplementedError

    def step(
        self, action: int
    ) -> tuple[Dict, SupportsFloat, bool, bool, dict[str, Any]]:
        """Send action to the game, hold it for action_repeat frames, return transition.

        During normal play the action selects a cardinal/diagonal direction or
        stay.  During the gameover animation (before the results screen appears)
        no action is sent and the episode is kept alive.  The episode terminates
        on the first results frame; the terminal reward is the survival time in
        seconds (OCR'd once and cached so repeated results frames don't re-run
        Tesseract).  Returns (obs, reward, terminated, truncated, info).
        With pixel_episodes=False the action is always sent and the reward is 0.
        """
        if not self.pixel_episodes:
            self._send_movement_action(action)
            self._wait_for_step_end()
            return self._get_obs(), 0.0, False, False, self._get_info()

        if self._phase == "gameover":
            # Release all held keys; do not press anything during the death animation.
            self._send_movement_action(0)  # action 0 = frozenset() = no keys
        elif self._phase != "results":
            self._send_movement_action(action)

        self._wait_for_step_end()
        obs = self._get_obs()  # updates self._phase
        truncated = False

        if self._phase == "results":
            if self._terminal_reward is None:
                secs = self._extract_survival_seconds(self._latest_frame)
                self._terminal_reward = float(secs) * self.reward_scale if secs is not None else 0.0
            reward = self._terminal_reward
            terminated = True
        elif self._phase == "gameover":
            reward = 0.0
            terminated = False  # stay alive until results screen appears
        else:
            reward = 0.0
            terminated = False

        info = self._get_info()
        if self._terminal_reward is not None:
            info["survived_seconds"] = round(self._terminal_reward / self.reward_scale)
        return obs, reward, terminated, truncated, info

    def _wait_for_step_end(self) -> None:
        """Sleep until this step's slot on the action_repeat / game_fps schedule ends."""
        period = self.action_repeat / self.game_fps
        now = time.monotonic()
        if self._step_due is None or now - self._step_due > period:
            if self._step_due is not None:
                self.late_steps += 1
            self._step_due = now + period
        else:
            self._step_due += period
        time.sleep(max(0.0, self._step_due - now))

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
        """Release held keys, the screen-capture handle, and the feature provider.

        A key left held through XTEST stays down after the env is gone and
        auto-repeats into whatever window has focus."""
        try:
            self._send_movement_action(0)
        except Exception:
            pass
        self._sct.close()
        if self._provider is not None:
            try:
                self._provider.close()
            except Exception:
                pass

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

        r_mean = float(frame[:, :, 0].mean())
        g_mean = float(frame[:, :, 1].mean())
        b_mean = float(frame[:, :, 2].mean())
        r_ratio = min(r_mean / (g_mean + eps), r_mean / (b_mean + eps))

        # Step 1 — bright overlay → levelup
        if center_mean > _LEVELUP_CENTER_MEAN_THRESHOLD:
            phase = "levelup"
        else:
            # Step 2 — high edge density → menu overlay.
            # Also require either high center brightness (card_sel A, levelup)
            # or very high r_ratio (red-tinted card_sel B) to avoid false
            # positives on dark-level gameplay/gameover frames.
            region = frame[_LEVELUP_ROW_SLICE, _LEVELUP_COL_SLICE]
            gray = _cv2.cvtColor(region, _cv2.COLOR_RGB2GRAY)
            edge_density = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())

            if edge_density > _MENU_EDGE_THRESHOLD and (
                center_mean > _MENU_CENTER_THRESHOLD
                or r_ratio > _CARD_RED_RATIO_THRESHOLD
            ):
                phase = "card_selection" if not self._gameplay_started else "levelup"
            else:
                # Step 3 — red tint → gameover or results
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

        obs = {"frames": np.stack(frames, axis=0)}  # (C, H, W)
        if self._provider is not None:
            obs["features"] = self._read_features()
        return obs

    def _get_info(self) -> Dict:
        """Return a diagnostic dict with hp, level, time, kills, and current game phase."""
        info = {"phase": self._phase}
        if self._provider is not None:
            info.update(self._feature_status)
        return info

    # ------------------------------------------------------------------
    # Feature provider
    # ------------------------------------------------------------------

    def _attach_provider(self) -> None:
        """Attach the provider to the game process; on failure record why and retry on reset()."""
        assert self._provider is not None and self._window is not None
        try:
            self._provider.attach(self._window.pid if self._window.pid > 0 else None)
            self._provider_attached = True
            self._provider_attach_error = None
        except Exception as e:
            self._provider_attached = False
            self._provider_attach_error = f"attach failed: {type(e).__name__}: {e}"

    def _read_features(self) -> dict[str, np.ndarray]:
        """Read one feature frame, never raising.

        Falls back to null_frame (valid=0) when the provider is unattached,
        returns None, raises, or returns a frame outside its declared space,
        and records the outcome in self._feature_status for info.
        """
        assert self._provider is not None and self._feature_space is not None
        t0 = time.perf_counter()
        frame, error = None, None
        if not self._provider_attached:
            error = self._provider_attach_error
        else:
            try:
                frame = self._provider.read()
                if frame is not None and not self._feature_space.contains(frame):
                    frame, error = None, "provider frame is outside its declared observation space"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

        if frame is None:
            features = null_frame(self._feature_space)
        else:
            features = {
                k: np.asarray(frame[k], dtype=sub.dtype) for k, sub in self._feature_space.spaces.items()
            }
        self._feature_status = {
            "features_ok": frame is not None,
            "features_read_ms": (time.perf_counter() - t0) * 1e3,
        }
        if error is not None:
            self._feature_status["features_error"] = error
        return features

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _extract_survival_seconds(self, frame: np.ndarray) -> Optional[int]:
        """OCR the 'Survived: MM:SS' value from a results-screen frame.

        Returns total seconds survived, or None if the text cannot be parsed
        (e.g. the frame is not actually a results screen).
        """
        y_sl, x_sl = _SURVIVED_TIME_ROI
        crop = frame[y_sl, x_sl]
        big = cv2.resize(
            crop,
            (0, 0),
            fx=_SURVIVED_OCR_SCALE,
            fy=_SURVIVED_OCR_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(big, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, _SURVIVED_THRESH, 255, cv2.THRESH_BINARY_INV)
        padded = cv2.copyMakeBorder(
            thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
        )
        text = pytesseract.image_to_string(
            padded,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:",
        ).strip()
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
        if not m:
            return None
        return int(m.group(1)) * 60 + int(m.group(2))

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
