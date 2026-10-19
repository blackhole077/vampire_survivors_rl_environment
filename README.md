# vampire_survivors_rl_environment

## Setup

```bash
uv sync                      # creates .venv (Python 3.12, from .python-version) with the env and pytest
source .venv/bin/activate
pytest tests
```

OCR needs the `tesseract` binary (`sudo apt install tesseract-ocr`).

## Optional game-state features

By default the env is pixels-only. You can optionally plug in a **feature provider**, which adds structured game state (mode, HP/XP/level/time, build, level-up offers) to each observation. No provider ships with or is required by this package. The contract lives in `vampire_survivors_env/features/spec.py`.

```python
env = VampireSurvivorsEnv(
    feature_provider="my_package.my_module:MyProvider",  # or an instance, or an entry-point name
    feature_provider_kwargs={...},
)
obs["frames"]    # unchanged
obs["features"]  # {"valid", "mode", "scalars", "weapons", "passives", "offers", [optional keys]}
info["features_ok"], info["features_read_ms"], info.get("features_error")
```

- **`feature_provider=None` (default):** observations and info are byte-for-byte what they were before; `tests/test_env_features.py` pins this.
- **Provider failures never crash an episode.** If `read()` returns `None`, raises, or returns a frame outside its declared space, `obs["features"]` becomes the null frame (`valid=0`, mode `UNKNOWN`, empty slots). The failure is recorded in `info`.
- **Incompatible providers are refused at construction.** A provider is incompatible if its spec version has a different major, or its space violates the spec.
- **Spec versioning:** `SPEC_VERSION` is `MAJOR.MINOR`. Minor bumps only add modes, vocabulary entries, or optional keys. Space sizes are fixed by capacities (`MODE_CAPACITY`, `VOCAB_CAPACITY`), so adding these never changes a shape.
- **Item vocabulary:** `vampire_survivors_env/features/vocab.json` maps item names to indices (0 = EMPTY, 1 = UNKNOWN). It is append-only, and `tests/data/vocab_lock.json` pins existing indices.

### Writing a provider

Subclass `vampire_survivors_env.features.FeatureProvider` (or duck-type it). Return `build_space(optional_keys)` from `observation_space()`, and build frames with `encode_scalars` / `encode_slots`. To check an implementation:

```bash
pytest tests/conformance --provider "my_package.my_module:MyProvider" \
    --provider-kwargs '{"some": "arg"}' [--provider-pid PID] [--read-budget-ms 1.0]
```

`vampire_survivors_env.features.mock.MockProvider` replays synthetic or recorded traces (`save_trace` / `load_trace`). Use it to develop consumers without the game.
