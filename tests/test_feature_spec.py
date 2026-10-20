"""Unit tests for the feature spec, vocabulary, loader, and mock provider.

These also prove the conformance checks bite: each broken provider or frame
below must be rejected.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from vampire_survivors_env.features.loader import load_provider
from vampire_survivors_env.features.mock import MockProvider, synthetic_trace
from vampire_survivors_env.features.spec import (
    MAX_ITEM_LEVEL,
    SPEC_VERSION,
    FeatureProvider,
    GameState,
    IncompatibleProviderError,
    ItemSlot,
    Mode,
    build_space,
    check_compatible,
    core_space,
    encode_scalars,
    encode_slots,
    frame_problems,
    null_frame,
)
from vampire_survivors_env.features.vocab import VOCAB_PATH, load_vocab, validate_vocab_doc

LOCK_PATH = Path(__file__).parent / "data" / "vocab_lock.json"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_vocab_file_is_valid():
    vocab = load_vocab()
    assert vocab.key(0) == "EMPTY" and vocab.key(1) == "UNKNOWN"
    assert vocab.index("WHIP") >= 2
    assert vocab.index("NOT_AN_ITEM") == 1
    assert vocab.index(None) == 0


def test_vocab_is_append_only():
    locked = json.loads(LOCK_PATH.read_text())["keys"]
    current = [e.key for e in load_vocab().entries]
    changed = [(i, k, current[i] if i < len(current) else None) for i, k in enumerate(locked) if i >= len(current) or current[i] != k]
    assert not changed, f"vocab indices changed (index, locked, now): {changed[:10]}"


def test_vocab_validation_rejects_renumbering():
    doc = json.loads(VOCAB_PATH.read_text())
    doc["entries"][5]["index"] = 99
    with pytest.raises(ValueError, match="contiguous"):
        validate_vocab_doc(doc)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def test_encode_scalars_clips_and_handles_bad_values():
    s = encode_scalars(stage_time_s=10_000, hp=150, max_hp=100, level=5000, xp_frac=float("nan"))
    assert core_space()["scalars"].contains(s)
    assert s[0] == 2.0 and s[1] == 1.0 and s[3] == 10.0 and s[4] == 0.0
    assert encode_scalars(0, 10, 0, 1, 0)[1] == 0.0  # max_hp == 0


def test_encode_slots_packs_clips_and_truncates():
    out = encode_slots([(0, 0), (5, 12), (7, 0)] + [(9, 1)] * 10, 6)
    assert out.shape == (6, 2)
    assert out[0].tolist() == [5, MAX_ITEM_LEVEL]
    assert out[1].tolist() == [7, 1]  # non-empty items get level >= 1
    assert core_space()["weapons"].contains(out)


# ---------------------------------------------------------------------------
# Provider compatibility
# ---------------------------------------------------------------------------


class _Provider(FeatureProvider):
    def __init__(self, space=None, version=SPEC_VERSION):
        self._space = space if space is not None else core_space()
        self.spec_version = version

    def observation_space(self):
        return self._space

    def read(self):
        return None


def test_check_compatible_accepts_core_and_optional():
    check_compatible(_Provider(build_space(("enemies",))))


def test_check_compatible_rejects_major_mismatch():
    with pytest.raises(IncompatibleProviderError, match=r"implements feature spec 2\.0"):
        check_compatible(_Provider(version="2.0"))


def test_check_compatible_rejects_malformed_version():
    with pytest.raises(IncompatibleProviderError, match="MAJOR.MINOR"):
        check_compatible(_Provider(version="one"))


def test_check_compatible_rejects_missing_methods():
    class NotAProvider:
        spec_version = SPEC_VERSION

    with pytest.raises(IncompatibleProviderError, match="missing"):
        check_compatible(NotAProvider())


def test_check_compatible_rejects_changed_core_key():
    spaces = dict(core_space().spaces)
    spaces["mode"] = gym.spaces.Discrete(7)
    with pytest.raises(IncompatibleProviderError, match="core key 'mode'"):
        check_compatible(_Provider(gym.spaces.Dict(spaces)))


def test_check_compatible_rejects_missing_core_key():
    spaces = dict(core_space().spaces)
    del spaces["offers"]
    with pytest.raises(IncompatibleProviderError, match="missing core key 'offers'"):
        check_compatible(_Provider(gym.spaces.Dict(spaces)))


def test_extra_keys_need_prefix_unless_provider_is_newer_minor():
    spaces = dict(core_space().spaces)
    spaces["gold"] = gym.spaces.Box(0, 1, (1,))
    with pytest.raises(IncompatibleProviderError, match="'gold'"):
        check_compatible(_Provider(gym.spaces.Dict(spaces)))
    major, minor = SPEC_VERSION.split(".")
    check_compatible(_Provider(gym.spaces.Dict(spaces), version=f"{major}.{int(minor) + 1}"))
    spaces["x_gold"] = spaces.pop("gold")
    check_compatible(_Provider(gym.spaces.Dict(spaces)))


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------


def _good_frame():
    return next(f for f in synthetic_trace(seed=3) if f is not None and int(f["mode"]) == Mode.LEVELUP)


def test_synthetic_frames_conform():
    space = core_space()
    frames = [f for f in synthetic_trace(seed=3) if f is not None]
    assert {int(f["mode"]) for f in frames} >= {Mode.PLAYING, Mode.LEVELUP, Mode.GAMEOVER, Mode.RESULTS}
    for f in frames:
        assert frame_problems(f, space) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda f: f.update(scalars=f["scalars"] + 5), "outside"),
        (lambda f: f.update(mode=np.int64(9)), "not a Mode"),
        (lambda f: f.update(valid=np.int64(0)), "valid=1"),
        (lambda f: f.pop("offers"), "frame keys"),
        (lambda f: f.update(mode=np.int64(Mode.PLAYING)), "offers must be EMPTY"),
        (lambda f: f["weapons"].__setitem__(5, (0, 3)), "EMPTY slots must have level 0"),
        (lambda f: f["weapons"].__setitem__(0, (0, 0)), "after filled"),
        (lambda f: f["passives"].__setitem__(0, (3, 0)), "level >= 1"),
    ],
)
def test_frame_problems_catch_broken_frames(mutate, expected):
    f = {k: np.array(v, copy=True) for k, v in _good_frame().items()}
    f["weapons"][:2] = [(2, 3), (3, 1)]
    mutate(f)
    problems = frame_problems(f, core_space())
    assert any(expected in p for p in problems), problems


def test_null_frame_is_in_space_and_marked_invalid():
    space = build_space(("enemies",))
    f = null_frame(space)
    assert space.contains(f)
    assert int(f["valid"]) == 0 and int(f["mode"]) == Mode.UNKNOWN
    assert (f["weapons"] == 0).all()


# ---------------------------------------------------------------------------
# Loader & mock
# ---------------------------------------------------------------------------


def test_loader_import_path_instance_and_errors():
    p = load_provider("vampire_survivors_env.features.mock:MockProvider", {"seed": 2})
    assert isinstance(p, MockProvider)
    assert load_provider(p) is p
    with pytest.raises(ValueError):
        load_provider(p, {"seed": 1})
    with pytest.raises(LookupError, match="entry point"):
        load_provider("no_such_provider_entry_point")
    with pytest.raises(ModuleNotFoundError):
        load_provider("no_such_module:Thing")


def test_mock_replays_none_and_raise_steps():
    frame = {k: v for k, v in null_frame(core_space()).items()}
    frame["valid"] = np.int64(1)
    p = MockProvider(trace=[None, {"raise": "boom"}, frame], loop=False)
    assert p.read() is None
    with pytest.raises(RuntimeError, match="boom"):
        p.read()
    assert int(p.read()["valid"]) == 1
    assert p.read() is None  # exhausted, loop=False


def test_game_state_encodes_to_a_conforming_frame():
    vocab = load_vocab()
    whip, spinach = vocab.index("WHIP"), vocab.index("SPINACH")
    state = GameState(
        mode=Mode.LEVELUP, stage_time_s=900.0, hp=50.0, max_hp=200.0, level=12, xp_frac=0.25,
        weapons=[ItemSlot(whip, 3)], passives=[ItemSlot(spinach, 1)],
        offers=[ItemSlot(whip, 4), ItemSlot(spinach, 2), ItemSlot(vocab.index("NOT_AN_ITEM"), 1)],
    )
    frame = state.encode()
    assert frame_problems(frame, build_space()) == []
    np.testing.assert_allclose(frame["scalars"], encode_scalars(900.0, 50.0, 200.0, 12, 0.25))
    assert frame["offers"].tolist() == [[whip, 4], [spinach, 2], [1, 1], [0, 0]]  # unmapped -> UNKNOWN


def test_game_state_carries_optional_keys():
    space = build_space(("enemies",))
    enemies = np.zeros(space["enemies"].shape, dtype=np.float32)
    frame = GameState(Mode.PLAYING, 0.0, 100.0, 100.0, 1, 0.0, extras={"enemies": enemies}).encode()
    assert frame_problems(frame, space) == []
