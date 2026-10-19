"""Env integration tests for the optional feature provider.  No game needed:
tests/fake_game.py stands in for X11 and screen capture.
"""

import json

import numpy as np
import pytest

from fake_game import FAKE_PID, install_fake_game
from make_env_golden import ACTIONS, GOLDEN_PATH, obs_digest, record_run
from vampire_survivors_env.env import VampireSurvivorsEnv
from vampire_survivors_env.features.mock import MockProvider
from vampire_survivors_env.features.spec import IncompatibleProviderError, Mode, core_space, null_frame

MOCK_PATH = "vampire_survivors_env.features.mock:MockProvider"


@pytest.fixture
def screen(monkeypatch):
    return install_fake_game(monkeypatch)


def make_env(**kwargs) -> VampireSurvivorsEnv:
    return VampireSurvivorsEnv(frame_stack=4, resolution=(100, 100), **kwargs)


def _valid_frame(mode: Mode = Mode.PLAYING) -> dict:
    f = null_frame(core_space())
    f["valid"] = np.int64(1)
    f["mode"] = np.int64(mode)
    return f


# ---------------------------------------------------------------------------
# provider=None is unchanged
# ---------------------------------------------------------------------------


def test_no_provider_matches_pre_integration_golden(screen):
    golden = json.loads(GOLDEN_PATH.read_text())
    env = make_env()
    assert record_run(env) == golden
    env.close()


def test_no_provider_obs_and_info_have_no_feature_keys(screen):
    env = make_env(feature_provider=None)
    obs, info = env.reset()
    assert set(obs) == {"frames"} and set(env.observation_space.spaces) == {"frames"}
    _, _, _, _, info = env.step(0)
    assert set(info) == {"phase"}


def test_provider_does_not_change_pixel_frames(screen):
    golden = json.loads(GOLDEN_PATH.read_text())
    env = make_env(feature_provider=MOCK_PATH, feature_provider_kwargs={"seed": 5})
    obs, _ = env.reset()
    digests = [obs_digest({"frames": obs["frames"]})]
    for a in ACTIONS:
        obs, *_ = env.step(a)
        digests.append(obs_digest({"frames": obs["frames"]}))
    assert digests == [s["obs"] for s in golden["steps"]]


# ---------------------------------------------------------------------------
# provider active
# ---------------------------------------------------------------------------


def test_features_in_obs_and_space(screen):
    provider = MockProvider(optional_keys=("enemies",))
    env = make_env(feature_provider=provider)
    assert provider.attached_pid == FAKE_PID
    assert set(env.observation_space.spaces) == {"frames", "features"}
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert "enemies" in obs["features"]
    for _ in range(30):
        obs, _, _, _, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert {"phase", "features_ok", "features_read_ms"} <= set(info)
    env.close()
    assert provider.closed


def test_incompatible_provider_is_refused_at_construction(screen):
    with pytest.raises(IncompatibleProviderError, match="2.0"):
        make_env(feature_provider=MockProvider(spec_version="2.0"))


def test_bad_import_path_is_refused_at_construction(screen):
    with pytest.raises(ModuleNotFoundError):
        make_env(feature_provider="not_a_real_module:Provider")


# ---------------------------------------------------------------------------
# provider failures never break the episode
# ---------------------------------------------------------------------------


def test_failing_provider_falls_back_to_null_features(screen):
    bad_shape = _valid_frame()
    bad_shape["scalars"] = np.full(5, 99.0, dtype=np.float32)
    trace = [_valid_frame(), None, {"raise": "segfault-ish"}, bad_shape, _valid_frame()]
    env = make_env(feature_provider=MockProvider(trace=trace))
    obs, info = env.reset()  # consumes trace[0]
    assert info["features_ok"] and int(obs["features"]["valid"]) == 1

    results = []
    for _ in range(len(trace) - 1):
        obs, _, terminated, _, info = env.step(0)
        assert env.observation_space.contains(obs)
        results.append((info["features_ok"], info.get("features_error"), int(obs["features"]["valid"])))

    assert results[0] == (False, None, 0)  # read() -> None: not an error
    assert results[1][0] is False and "segfault-ish" in results[1][1] and results[1][2] == 0
    assert results[2][0] is False and "outside" in results[2][1] and results[2][2] == 0
    assert results[3] == (True, None, 1)  # recovers


def test_attach_failure_is_recorded_and_retried_on_reset(screen):
    class FlakyAttach(MockProvider):
        def attach(self, pid):
            super().attach(pid)
            if self.n_attach == 1:
                raise PermissionError("ptrace denied")

    provider = FlakyAttach()
    env = make_env(feature_provider=provider)  # first attach fails, no exception
    _, _, _, _, info = env.step(0)
    assert info["features_ok"] is False and "ptrace denied" in info["features_error"]
    _, info = env.reset()  # retries attach
    assert provider.n_attach == 2
    assert "features_error" not in info
    oks = [env.step(0)[4]["features_ok"] for _ in range(5)]  # synthetic trace starts with 3 'loading' Nones
    assert oks[-1] is True


def test_close_survives_provider_close_error(screen):
    class BadClose(MockProvider):
        def close(self):
            raise OSError("already gone")

    env = make_env(feature_provider=BadClose())
    env.close()


# ---------------------------------------------------------------------------
# Timing budget
# ---------------------------------------------------------------------------


def test_feature_overhead_per_step_within_budget(screen):
    """Env-side cost of the feature path (read + validation + copy) per step, with the mock."""
    env = make_env(feature_provider=MockProvider(optional_keys=("enemies",)))
    env.reset()
    ms = []
    for _ in range(300):
        _, _, _, _, info = env.step(0)
        ms.append(info["features_read_ms"])
    ms = np.array(ms)
    print(f"feature path ms/step: mean={ms.mean():.4f} p99={np.percentile(ms, 99):.4f} max={ms.max():.4f}")
    assert ms.mean() < 1.0
