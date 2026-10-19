"""Conformance suite for FeatureProvider implementations.

Runs against the mock providers by default; point it at your own with
--provider (see tests/conftest.py).  A provider passes if it is compatible
with SPEC_VERSION, every frame it returns conforms to its declared space and
the spec's semantic rules, it never raises on ordinary reads, and reads fit
the time budget.
"""

import time

import numpy as np
import pytest

from vampire_survivors_env.features.loader import load_provider
from vampire_survivors_env.features.spec import (
    SLOT_KEYS,
    SPEC_VERSION,
    Mode,
    check_compatible,
    frame_problems,
    null_frame,
    space_problems,
)
from vampire_survivors_env.features.vocab import load_vocab


@pytest.fixture
def provider(provider_config, request):
    path, kwargs = provider_config
    p = load_provider(path, dict(kwargs))
    yield p
    p.close()


@pytest.fixture
def attached(provider, request):
    provider.attach(request.config.getoption("--provider-pid"))
    return provider


@pytest.fixture
def frames(attached, request):
    """(frames, n_none) from --conformance-reads consecutive reads."""
    n = request.config.getoption("--conformance-reads")
    out, n_none = [], 0
    for _ in range(n):
        f = attached.read()
        if f is None:
            n_none += 1
        else:
            out.append(f)
    return out, n_none


def test_spec_version_is_compatible(provider):
    assert int(provider.spec_version.split(".")[0]) == int(SPEC_VERSION.split(".")[0])
    check_compatible(provider)


def test_observation_space_is_stable_and_available_before_attach(provider, request):
    before = provider.observation_space()
    assert space_problems(before, provider.spec_version) == []
    provider.attach(request.config.getoption("--provider-pid"))
    assert provider.observation_space() == before


def test_null_frame_fits_declared_space(provider):
    space = provider.observation_space()
    assert space.contains(null_frame(space))


def test_reads_return_none_or_conforming_frames(attached, frames):
    space = attached.observation_space()
    got, _ = frames
    for i, f in enumerate(got):
        problems = frame_problems(f, space)
        assert not problems, f"frame {i}: {problems}"


def test_some_reads_succeed(frames):
    got, n_none = frames
    assert got, f"all {n_none} reads returned None — is a run active?"


def test_mode_values_are_spec_modes(frames):
    got, _ = frames
    modes = {int(f["mode"]) for f in got}
    assert modes <= {m.value for m in Mode}
    print("modes seen:", sorted(Mode(m).name for m in modes))


def test_item_indices_are_in_vocabulary(frames):
    vocab = load_vocab()
    got, _ = frames
    for f in got:
        for key in SLOT_KEYS:
            idx = np.asarray(f[key])[:, 0]
            assert idx.max() < len(vocab), f"{key} index {idx.max()} beyond vocab size {len(vocab)}"
            deprecated = [i for i in idx if vocab.entries[i].deprecated]
            assert not deprecated, f"{key} uses deprecated vocab entries {deprecated}"


def test_offers_only_during_levelup(frames):
    got, _ = frames
    for f in got:
        if int(f["mode"]) != Mode.LEVELUP:
            assert (np.asarray(f["offers"]) == 0).all()


def _next_frame(provider, tries: int = 50):
    for _ in range(tries):
        f = provider.read()
        if f is not None:
            return f
    return None


def test_frames_are_not_aliased_between_reads(attached):
    """Each read() must return fresh arrays: the env and replay buffers keep earlier frames."""
    first, second = _next_frame(attached), _next_frame(attached)
    if first is None or second is None:
        pytest.skip("provider returned fewer than two frames")
    for k in first:
        assert not np.shares_memory(first[k], second[k]), f"{k!r} array reused across reads"


def test_read_time_within_budget(attached, request):
    n = request.config.getoption("--conformance-reads")
    budget_ms = request.config.getoption("--read-budget-ms")
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        attached.read()
        times[i] = time.perf_counter() - t0
    ms = times * 1e3
    print(f"read() ms: mean={ms.mean():.4f} p50={np.median(ms):.4f} p99={np.percentile(ms, 99):.4f} max={ms.max():.4f}")
    assert ms.mean() <= budget_ms, f"mean read() {ms.mean():.3f} ms exceeds {budget_ms} ms"


def test_close_is_idempotent(provider_config):
    path, kwargs = provider_config
    p = load_provider(path, dict(kwargs))
    p.close()
    p.close()
