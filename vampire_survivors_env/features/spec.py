"""Game-state feature provider spec, version SPEC_VERSION.

A feature provider is an optional, structured side-channel next to the pixel
observations: game mode, a few normalised scalars, the current build, and the
level-up offers.  The env never imports a concrete provider; it loads one from
config (see loader.py) and checks it against this module.

Versioning
  SPEC_VERSION is "MAJOR.MINOR".  A provider is compatible iff its MAJOR
  matches.  MINOR bumps may only add: new Mode values (within MODE_CAPACITY),
  new vocabulary entries (within VOCAB_CAPACITY), or new OPTIONAL_KEYS.  Space
  sizes are fixed by the capacities, so none of these change a space's shape.

Frame layout (a FeatureFrame is dict[str, np.ndarray] matching the space)
  valid     Discrete(2)       1 on every frame a provider returns; the env
                              substitutes null_frame() (valid=0) when read()
                              returns None or fails.
  mode      Discrete(16)      a Mode value; consumers treat values they do not
                              know as Mode.OTHER.
  scalars   Box(5,) float32   see SCALAR_NAMES / encode_scalars().
  weapons   MultiDiscrete     (N_WEAPON_SLOTS, 2): (vocab index, level).
  passives  MultiDiscrete     (N_PASSIVE_SLOTS, 2): same encoding.
  offers    MultiDiscrete     (N_OFFER_SLOTS, 2): (vocab index, level the
                              item would have after picking it, 1 = new).
                              All slots EMPTY unless mode == LEVELUP.
  Slot encoding: (EMPTY_INDEX, 0) for an empty slot; otherwise a vocab index
  >= 1 and a level in 1..MAX_ITEM_LEVEL (higher levels are clipped).  Items
  the provider cannot map use UNKNOWN_INDEX.  Slots are filled in game order
  with empties last.

Optional keys
  Anything outside CORE_KEYS must either be registered in OPTIONAL_KEYS (and
  then match that space exactly) or start with "x_" (provider-specific).
  Consumers detect optional keys with `key in provider_space.spaces`.

Writing game state is deliberately not part of this interface.
"""

import abc
from enum import IntEnum
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

SPEC_VERSION = "1.0"

FeatureFrame = dict[str, np.ndarray]


class IncompatibleProviderError(ValueError):
    """Raised when a provider's version or observation space violates the spec."""


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


class Mode(IntEnum):
    """What the game is showing.  Append-only; new values need a MINOR bump."""

    UNKNOWN = 0
    PLAYING = 1
    LEVELUP = 2
    CHEST = 3
    GAMEOVER = 4
    RESULTS = 5
    OTHER = 6  # any other menu/overlay: pause, arcana select, new character, ...


MODE_CAPACITY = 16  # Discrete size; values >= len(Mode) are reserved

# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

SCALAR_NAMES = ("stage_time", "hp_frac", "max_hp", "level", "xp_frac")

STAGE_TIME_SCALE_S = 1800.0  # 30:00 stage clock -> 1.0
MAX_HP_SCALE = 500.0
LEVEL_SCALE = 100.0

_SCALAR_LOW = np.zeros(len(SCALAR_NAMES), dtype=np.float32)
_SCALAR_HIGH = np.array([2.0, 1.0, 4.0, 10.0, 1.0], dtype=np.float32)


def encode_scalars(
    stage_time_s: float, hp: float, max_hp: float, level: int, xp_frac: float
) -> np.ndarray:
    """Normalise and clip raw game values into the `scalars` vector.

    xp_frac is progress toward the next level in [0, 1].  hp_frac is
    hp / max_hp, or 0 when max_hp is not positive.
    """
    hp_frac = hp / max_hp if max_hp > 0 else 0.0
    raw = np.array(
        [
            stage_time_s / STAGE_TIME_SCALE_S,
            hp_frac,
            max_hp / MAX_HP_SCALE,
            level / LEVEL_SCALE,
            xp_frac,
        ],
        dtype=np.float32,
    )
    return np.clip(np.nan_to_num(raw), _SCALAR_LOW, _SCALAR_HIGH)


# ---------------------------------------------------------------------------
# Item slots
# ---------------------------------------------------------------------------

N_WEAPON_SLOTS = 6
N_PASSIVE_SLOTS = 6
N_OFFER_SLOTS = 4
MAX_ITEM_LEVEL = 8
VOCAB_CAPACITY = 1024  # MultiDiscrete size for item indices; see vocab.json

EMPTY_INDEX = 0
UNKNOWN_INDEX = 1

SLOT_KEYS = ("weapons", "passives", "offers")


def _slot_space(n_slots: int) -> gym.spaces.MultiDiscrete:
    nvec = np.tile(np.array([VOCAB_CAPACITY, MAX_ITEM_LEVEL + 1]), (n_slots, 1))
    return gym.spaces.MultiDiscrete(nvec)


def encode_slots(items: list[tuple[int, int]], n_slots: int) -> np.ndarray:
    """Pack [(vocab_index, level), ...] into an (n_slots, 2) array.

    Extra items beyond n_slots are dropped, missing slots are EMPTY, levels
    are clipped to 1..MAX_ITEM_LEVEL for non-empty items.
    """
    out = np.zeros((n_slots, 2), dtype=np.int64)
    for i, (idx, level) in enumerate(items[:n_slots]):
        if idx == EMPTY_INDEX:
            continue
        out[i] = (idx, min(max(int(level), 1), MAX_ITEM_LEVEL))
    # Keep empties last even if the caller passed EMPTY placeholders.
    filled = out[out[:, 0] != EMPTY_INDEX]
    out[:] = 0
    out[: len(filled)] = filled
    return out


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def core_space() -> gym.spaces.Dict:
    """The required keys every provider's observation_space() must contain."""
    return gym.spaces.Dict(
        {
            "valid": gym.spaces.Discrete(2),
            "mode": gym.spaces.Discrete(MODE_CAPACITY),
            "scalars": gym.spaces.Box(_SCALAR_LOW, _SCALAR_HIGH, dtype=np.float32),
            "weapons": _slot_space(N_WEAPON_SLOTS),
            "passives": _slot_space(N_PASSIVE_SLOTS),
            "offers": _slot_space(N_OFFER_SLOTS),
        }
    )


CORE_KEYS = frozenset(core_space().spaces)

N_ENEMY_SLOTS = 16
ENEMY_POS_SCALE = 400.0  # world units; offsets are divided by this then clipped to [-1, 1]

# Registered optional keys.  A provider that emits one of these must use
# exactly this space.  Add new entries with a MINOR bump.
OPTIONAL_KEYS: dict[str, Callable[[], gym.spaces.Space]] = {
    # Nearest enemies first: (dx, dy, present).  dx/dy are offsets from the
    # player / ENEMY_POS_SCALE, clipped to [-1, 1]; unused rows are all zero.
    "enemies": lambda: gym.spaces.Box(
        low=np.tile(np.array([-1.0, -1.0, 0.0], dtype=np.float32), (N_ENEMY_SLOTS, 1)),
        high=np.ones((N_ENEMY_SLOTS, 3), dtype=np.float32),
        dtype=np.float32,
    ),
}

EXTENSION_PREFIX = "x_"


def build_space(optional_keys: tuple[str, ...] = ()) -> gym.spaces.Dict:
    """core_space() plus the given registered optional keys."""
    spaces = dict(core_space().spaces)
    for key in optional_keys:
        if key not in OPTIONAL_KEYS:
            raise KeyError(f"{key!r} is not a registered optional key")
        spaces[key] = OPTIONAL_KEYS[key]()
    return gym.spaces.Dict(spaces)


def null_frame(space: gym.spaces.Dict) -> FeatureFrame:
    """The frame used when no provider data is available: valid=0, mode UNKNOWN,
    zero scalars, all slots EMPTY, optional keys zeroed (clipped into bounds)."""
    out: FeatureFrame = {}
    for key, sub in space.spaces.items():
        if isinstance(sub, gym.spaces.Discrete):
            out[key] = np.asarray(sub.start, dtype=sub.dtype)
        elif isinstance(sub, gym.spaces.MultiDiscrete):
            out[key] = np.array(sub.start, dtype=sub.dtype, copy=True)
        elif isinstance(sub, gym.spaces.Box):
            out[key] = np.clip(np.zeros(sub.shape, dtype=sub.dtype), sub.low, sub.high)
        elif isinstance(sub, gym.spaces.MultiBinary):
            out[key] = np.zeros(sub.shape, dtype=sub.dtype)
        else:
            raise TypeError(f"null_frame: unsupported space for {key!r}: {sub}")
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _parse_version(version: str) -> tuple[int, int]:
    major, minor = str(version).split(".")
    return int(major), int(minor)


def space_problems(space: gym.spaces.Space, provider_version: str = SPEC_VERSION) -> list[str]:
    """Return every way `space` violates the spec (empty list if it conforms).

    Unrecognised keys are tolerated when provider_version has a newer MINOR
    than SPEC_VERSION, since they may be optional keys registered after us.
    """
    newer_minor = _parse_version(provider_version)[1] > _parse_version(SPEC_VERSION)[1]
    if not isinstance(space, gym.spaces.Dict):
        return [f"observation_space() must be a gymnasium.spaces.Dict, got {type(space).__name__}"]
    problems = []
    for key, expected in core_space().spaces.items():
        if key not in space.spaces:
            problems.append(f"missing core key {key!r}")
        elif space.spaces[key] != expected:
            problems.append(f"core key {key!r} is {space.spaces[key]}, spec requires {expected}")
    for key, sub in space.spaces.items():
        if key in CORE_KEYS:
            continue
        if key in OPTIONAL_KEYS:
            if sub != OPTIONAL_KEYS[key]():
                problems.append(f"optional key {key!r} is {sub}, spec requires {OPTIONAL_KEYS[key]()}")
        elif not key.startswith(EXTENSION_PREFIX) and not newer_minor:
            problems.append(
                f"key {key!r} is neither core, registered optional, nor prefixed {EXTENSION_PREFIX!r}"
            )
    return problems


def frame_problems(frame: FeatureFrame, space: gym.spaces.Dict) -> list[str]:
    """Return every way a frame returned by read() violates the spec."""
    if not isinstance(frame, dict):
        return [f"read() must return a dict or None, got {type(frame).__name__}"]
    if set(frame) != set(space.spaces):
        return [f"frame keys {sorted(frame)} != space keys {sorted(space.spaces)}"]
    problems = [
        f"{key!r}={np.asarray(frame[key])!r} is outside {sub}"
        for key, sub in space.spaces.items()
        if not sub.contains(np.asarray(frame[key], dtype=sub.dtype))
        or np.asarray(frame[key]).shape != sub.shape
    ]
    if problems:
        return problems
    if int(frame["valid"]) != 1:
        problems.append("provider frames must have valid=1 (valid=0 is reserved for null_frame)")
    if int(frame["mode"]) not in Mode._value2member_map_:
        problems.append(f"mode {int(frame['mode'])} is not a Mode in spec {SPEC_VERSION}")
    for key in SLOT_KEYS:
        slots = np.asarray(frame[key])
        empty = slots[:, 0] == EMPTY_INDEX
        if np.any(slots[empty, 1] != 0):
            problems.append(f"{key}: EMPTY slots must have level 0")
        if np.any(slots[~empty, 1] < 1):
            problems.append(f"{key}: non-empty slots must have level >= 1")
        if empty.any() and not empty[np.argmax(empty):].all():
            problems.append(f"{key}: EMPTY slots must come after filled slots")
    if int(frame["mode"]) != Mode.LEVELUP and np.any(np.asarray(frame["offers"])[:, 0] != EMPTY_INDEX):
        problems.append("offers must be EMPTY unless mode == LEVELUP")
    return problems


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class FeatureProvider(abc.ABC):
    """Base class for game-state feature providers.

    Subclassing is optional: the env duck-types on these members, so any object
    with spec_version / observation_space / attach / read / close works.

    Lifecycle: the env calls attach() once after finding the game window (and
    again on reset() if attach failed), read() once per observation, and
    close() when the env closes.
    """

    spec_version: str = SPEC_VERSION

    @abc.abstractmethod
    def observation_space(self) -> gym.spaces.Dict:
        """Return the provider's feature space: build_space(...) plus any x_ keys.

        Must be constant for the provider's lifetime and callable before attach().
        """

    def attach(self, pid: Optional[int]) -> None:
        """Connect to the game.  pid is the game window's process ID, if known.

        Raise on failure; the env records the error and retries on reset().
        """

    @abc.abstractmethod
    def read(self) -> Optional[FeatureFrame]:
        """Return the current FeatureFrame, or None if no run is active or the
        state is not readable yet.  Should be fast (well under 1 ms) and
        should not raise for ordinary not-ready states."""

    def close(self) -> None:
        """Release any resources.  Must be safe to call more than once."""


def check_compatible(provider: object) -> gym.spaces.Dict:
    """Validate a provider against this spec and return its observation space.

    Raises IncompatibleProviderError naming the problem.
    """
    missing = [m for m in ("spec_version", "observation_space", "attach", "read", "close") if not hasattr(provider, m)]
    if missing:
        raise IncompatibleProviderError(f"{type(provider).__name__} is missing {missing}")
    name = type(provider).__name__
    try:
        theirs = _parse_version(provider.spec_version)
    except (ValueError, TypeError):
        raise IncompatibleProviderError(
            f"{name}.spec_version={provider.spec_version!r} is not 'MAJOR.MINOR'"
        ) from None
    ours = _parse_version(SPEC_VERSION)
    if theirs[0] != ours[0]:
        raise IncompatibleProviderError(
            f"{name} implements feature spec {provider.spec_version}, but this env "
            f"supports {ours[0]}.x (SPEC_VERSION={SPEC_VERSION}). Update the provider "
            f"or the env so the major versions match."
        )
    space = provider.observation_space()
    problems = space_problems(space, provider.spec_version)
    if problems:
        raise IncompatibleProviderError(f"{name}.observation_space() violates the spec: " + "; ".join(problems))
    return space
