"""Record golden hashes of pixel-only observations for test_env_features.py.

The golden file pins the observations produced with no feature provider, so
the provider=None path can be checked byte-for-byte.  It was generated from
env.py *before* the feature-provider integration landed.  Regenerate it only
when a pixel-pipeline change is intended:

    python tests/make_env_golden.py
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_game import install_fake_game  # noqa: E402
from vampire_survivors_env.env import VampireSurvivorsEnv  # noqa: E402

GOLDEN_PATH = ROOT / "tests" / "data" / "env_obs_golden.json"
ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 3]


def obs_digest(obs: dict) -> dict[str, str]:
    """sha256 of each obs array's dtype, shape, and raw bytes."""
    out = {}
    for key in sorted(obs):
        arr = obs[key]
        h = hashlib.sha256(f"{arr.dtype}|{arr.shape}|".encode())
        h.update(arr.tobytes())
        out[key] = h.hexdigest()
    return out


def record_run(env: VampireSurvivorsEnv) -> dict:
    """Reset then step through ACTIONS, recording obs digests and info keys."""
    obs, info = env.reset()
    steps = [{"obs": obs_digest(obs), "info_keys": sorted(info)}]
    for a in ACTIONS:
        obs, reward, terminated, truncated, info = env.step(a)
        steps.append(
            {
                "obs": obs_digest(obs),
                "info_keys": sorted(info),
                "reward": float(reward),
                "terminated": terminated,
            }
        )
    return {"observation_space": repr(env.observation_space), "steps": steps}


def main() -> None:
    mp = pytest.MonkeyPatch()
    try:
        install_fake_game(mp)
        env = VampireSurvivorsEnv(frame_stack=4, resolution=(100, 100))
        golden = record_run(env)
        env.close()
    finally:
        mp.undo()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1) + "\n")
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
