"""The game-state feature spec, version SPEC_VERSION.

The env sees the game as pixels.  A feature provider adds what the pixels
make hard to read: what the game is showing, the clock, health, level and
experience, the items the character owns, and the choices on a level-up
screen.  It is optional.  The env never imports a provider; it loads one by
name (loader.py) and checks it against this module.

Start with GameState.  A provider reads the game, fills in a GameState in
the game's own units, and calls encode().  Everything below GameState in
this file describes what encode() produces, so a provider that uses it never
needs the details.

The frame
  Each read() returns a FeatureFrame: a dict of numpy arrays that matches
  build_space().

  valid     1 on every frame a provider returns.  The env uses 0 for the
            null frame it substitutes when read() returns None or fails.
  mode      A Mode value.  Code that meets a value it does not know should
            treat it as Mode.OTHER.
  scalars   Five floats: stage time, hp fraction, max hp, level, xp
            fraction, each scaled to a small range (see encode_scalars).
  weapons   One row per slot: (vocabulary index, level).
  passives  The same, for passive items.
  offers    The same, for level-up choices, where level is the level the
            item would reach if picked (1 for a new item).  All empty
            unless mode is LEVELUP.

  An empty slot is (EMPTY_INDEX, 0), and empty slots come last.  An item the
  provider cannot name is UNKNOWN_INDEX.  Levels above MAX_ITEM_LEVEL are
  clipped.

Optional keys
  A provider may add keys beyond the core ones.  A key registered in
  OPTIONAL_KEYS must use exactly that space; any other key must start with
  "x_".  Consumers test for one with `key in space.spaces`.

Versions
  SPEC_VERSION is "MAJOR.MINOR", and a provider works with this env when the
  MAJOR numbers match.  A MINOR release may only add: Mode values, vocabulary
  entries or registered optional keys.  The spaces are sized by fixed
  capacities (MODE_CAPACITY, VOCAB_CAPACITY), so none of these change an
  array's shape.

The spec is read-only: nothing here writes to the game.
"""

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, NamedTuple, Optional

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


# ---------------------------------------------------------------------------
# GameState: what a provider reads, before encoding
# ---------------------------------------------------------------------------


class ItemSlot(NamedTuple):
    """One owned item, or one level-up offer.

    index is the item's vocabulary index (see vocab.py).  The provider maps the
    game's own item IDs onto it; an item it cannot map is UNKNOWN_INDEX.  level
    is the item's level now, or for an offer the level it would reach if
    picked (1 for an item not yet owned).
    """

    index: int
    level: int


@dataclass(frozen=True)
class GameState:
    """One reading of the game, in the game's own units.

    This is the contract between a provider and the spec.  A provider's job is
    to fill in these fields from wherever it gets them (process memory, a mod,
    a recording); encode() turns them into the FeatureFrame the policy sees,
    doing all the scaling, clipping and slot packing in one place.  A provider
    that builds its frames this way cannot get the layout wrong.

    Read it only while a run is on screen.  Between runs, in menus or while a
    stage loads, read() returns None and the env fills in null_frame().

    mode            What the game is showing (Mode).
    stage_time_s    The stage clock in seconds, as shown at the top of the
                    screen.  A stage normally ends at 1800 (30:00).
    hp, max_hp      Health now and its current maximum, in game HP.
    level           The character's level, starting at 1.
    xp_frac         Progress from this level to the next, 0 to 1.
    weapons         Owned weapons in the order the game lists them.
    passives        Owned passive items, likewise.
    offers          The level-up choices on screen.  Empty unless mode is
                    LEVELUP.
    extras          Registered optional keys (OPTIONAL_KEYS) and the
                    provider's own "x_" keys, already encoded to their spaces.

    Slot lists may be longer than the space allows; encode() keeps the first
    N_WEAPON_SLOTS, N_PASSIVE_SLOTS or N_OFFER_SLOTS.
    """

    mode: Mode
    stage_time_s: float
    hp: float
    max_hp: float
    level: int
    xp_frac: float
    weapons: Sequence[ItemSlot] = ()
    passives: Sequence[ItemSlot] = ()
    offers: Sequence[ItemSlot] = ()
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    def encode(self) -> FeatureFrame:
        """The FeatureFrame for this reading.  Check it with frame_problems() if in doubt."""
        frame: FeatureFrame = {
            "valid": np.int64(1),
            "mode": np.int64(self.mode),
            "scalars": encode_scalars(self.stage_time_s, self.hp, self.max_hp, int(self.level), self.xp_frac),
            "weapons": encode_slots(list(self.weapons), N_WEAPON_SLOTS),
            "passives": encode_slots(list(self.passives), N_PASSIVE_SLOTS),
            "offers": encode_slots(list(self.offers), N_OFFER_SLOTS),
        }
        frame.update(self.extras)
        return frame

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
