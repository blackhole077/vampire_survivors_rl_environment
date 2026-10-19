"""Trace-driven FeatureProvider for tests and for developing consumers without the game.

A trace is a list of steps; read() returns them in order (looping by default).
Each step is one of:
  None                      read() returns None (no run / not readable)
  {"raise": "message"}      read() raises RuntimeError(message)
  {key: value, ...}         a FeatureFrame; lists are converted to arrays

Traces can be generated (synthetic_trace) or saved/loaded as JSON
(save_trace / load_trace), e.g. to replay frames recorded from a real provider.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from vampire_survivors_env.features.spec import (
    N_ENEMY_SLOTS,
    N_OFFER_SLOTS,
    N_PASSIVE_SLOTS,
    N_WEAPON_SLOTS,
    SPEC_VERSION,
    FeatureFrame,
    FeatureProvider,
    Mode,
    build_space,
    encode_scalars,
    encode_slots,
)
from vampire_survivors_env.features.vocab import load_vocab

TraceStep = Optional[dict[str, Any]]


class MockProvider(FeatureProvider):
    """Replays a trace.  With no trace, replays synthetic_trace(seed=seed)."""

    def __init__(
        self,
        trace: Union[None, str, Path, list[TraceStep]] = None,
        optional_keys: tuple[str, ...] = (),
        loop: bool = True,
        seed: int = 0,
        spec_version: str = SPEC_VERSION,
    ) -> None:
        if trace is None:
            trace = synthetic_trace(seed=seed, optional_keys=tuple(optional_keys))
        elif isinstance(trace, (str, Path)):
            trace, file_keys = load_trace(trace)
            optional_keys = tuple(optional_keys) or file_keys
        self.spec_version = spec_version
        self._space = build_space(tuple(optional_keys))
        self._trace = [self._to_arrays(step) for step in trace]
        self._loop = loop
        self._pos = 0
        self.attached_pid: Optional[int] = None
        self.n_attach = 0
        self.closed = False

    def _to_arrays(self, step: TraceStep) -> TraceStep:
        if step is None or "raise" in step:
            return step
        return {
            k: np.asarray(v, dtype=self._space.spaces[k].dtype) if k in self._space.spaces else np.asarray(v)
            for k, v in step.items()
        }

    def observation_space(self):
        return self._space

    def attach(self, pid: Optional[int]) -> None:
        self.attached_pid = pid
        self.n_attach += 1

    def read(self) -> Optional[FeatureFrame]:
        if self._pos >= len(self._trace):
            if not self._loop:
                return None
            self._pos = 0
        step = self._trace[self._pos]
        self._pos += 1
        if step is None:
            return None
        if "raise" in step:
            raise RuntimeError(step["raise"])
        return {k: v.copy() for k, v in step.items()}

    def close(self) -> None:
        self.closed = True


def synthetic_trace(
    n_steps: int = 400, seed: int = 0, optional_keys: tuple[str, ...] = ()
) -> list[TraceStep]:
    """Simulate a plausible run: loading (None), play with level-ups and a
    chest, then game over, results, and a menu.  All frames conform to the spec."""
    rng = np.random.default_rng(seed)
    vocab = load_vocab()
    weapon_pool = vocab.indices("weapon")
    passive_pool = vocab.indices("passive")
    weapons: list[list[int]] = [[int(rng.choice(weapon_pool)), 1]]
    passives: list[list[int]] = []
    t, hp, max_hp, level, xp = 0.0, 100.0, 100.0, 1, 0.0

    def frame(mode: Mode, offers: list[tuple[int, int]] = ()) -> dict[str, Any]:
        f = {
            "valid": 1,
            "mode": int(mode),
            "scalars": encode_scalars(t, hp, max_hp, level, xp),
            "weapons": encode_slots([tuple(w) for w in weapons], N_WEAPON_SLOTS),
            "passives": encode_slots([tuple(p) for p in passives], N_PASSIVE_SLOTS),
            "offers": encode_slots(list(offers), N_OFFER_SLOTS),
        }
        if "enemies" in optional_keys:
            n = int(rng.integers(0, N_ENEMY_SLOTS + 1))
            enemies = np.zeros((N_ENEMY_SLOTS, 3), dtype=np.float32)
            pos = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
            order = np.argsort(np.linalg.norm(pos, axis=1))
            enemies[:n, :2] = pos[order]
            enemies[:n, 2] = 1.0
            f["enemies"] = enemies
        return f

    def make_offers() -> list[tuple[int, int]]:
        offers = []
        for slots, pool, cap in ((weapons, weapon_pool, 8), (passives, passive_pool, 5)):
            for idx, lvl in slots:
                if lvl < cap:
                    offers.append((idx, lvl + 1))
            if len(slots) < 6:
                owned = {s[0] for s in slots}
                offers.append((int(rng.choice([i for i in pool if i not in owned])), 1))
        rng.shuffle(offers)
        return offers[: int(rng.integers(3, N_OFFER_SLOTS + 1))]

    def take(offer: tuple[int, int]) -> None:
        idx, lvl = offer
        for slots in (weapons, passives):
            for s in slots:
                if s[0] == idx:
                    s[1] = lvl
                    return
        (weapons if idx in weapon_pool else passives).append([idx, lvl])

    trace: list[TraceStep] = [None] * 3  # run still loading
    death_at = n_steps - 12
    while len(trace) < death_at:
        t += 0.5
        xp += float(rng.uniform(0.02, 0.12))
        hp = float(np.clip(hp - rng.uniform(-1.0, 2.0), 1.0, max_hp))
        if xp >= 1.0:
            xp, level = 0.0, level + 1
            offers = make_offers()
            trace += [frame(Mode.LEVELUP, offers)] * 3
            if offers:
                take(offers[0])
            if level % 3 == 0:
                max_hp += 10.0
        elif rng.random() < 0.01:
            trace += [frame(Mode.CHEST)] * 2
        else:
            trace.append(frame(Mode.PLAYING))
    trace = trace[:death_at]
    hp = 0.0
    trace += [frame(Mode.GAMEOVER)] * 4 + [frame(Mode.RESULTS)] * 4 + [frame(Mode.OTHER)] * 2
    return [None if s is None else {k: np.asarray(v) for k, v in s.items()} for s in trace]


def save_trace(path: Union[str, Path], trace: list[TraceStep], optional_keys: tuple[str, ...] = ()) -> None:
    """Write a trace as JSON (arrays become nested lists)."""
    steps = [None if s is None else {k: np.asarray(v).tolist() for k, v in s.items()} for s in trace]
    doc = {"spec_version": SPEC_VERSION, "optional_keys": list(optional_keys), "steps": steps}
    Path(path).write_text(json.dumps(doc) + "\n")


def load_trace(path: Union[str, Path]) -> tuple[list[TraceStep], tuple[str, ...]]:
    """Read a trace written by save_trace; returns (steps, optional_keys)."""
    doc = json.loads(Path(path).read_text())
    return doc["steps"], tuple(doc.get("optional_keys", ()))
