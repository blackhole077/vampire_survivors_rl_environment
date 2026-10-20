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

A provider does one thing: it reads the game and fills in a `GameState`. That dataclass, in `spec.py`, is the interface. It holds the values in the game's own units, and its `encode()` method does all the scaling, clipping and slot packing. A provider that builds frames this way cannot get the layout wrong.

| Field | What it holds |
|---|---|
| `mode` | What the game is showing: a `Mode` (`PLAYING`, `LEVELUP`, `CHEST`, `GAMEOVER`, `RESULTS`, `OTHER`). |
| `stage_time_s` | The stage clock in seconds. A stage normally ends at 1800 (30:00). |
| `hp`, `max_hp` | Health now and its current maximum, in game HP. |
| `level` | The character's level, starting at 1. |
| `xp_frac` | Progress from this level to the next, 0 to 1. |
| `weapons`, `passives` | Owned items, in the game's order, as `ItemSlot(index, level)`. |
| `offers` | The level-up choices on screen, as `ItemSlot(index, level if picked)`. Empty unless `mode` is `LEVELUP`. |
| `extras` | Optional keys (`enemies`, `enemy_grid`) and your own `x_` keys, already encoded. |

`index` is the item's position in `vocab.json`. Map the game's item IDs onto it with `load_vocab().index(key)`; anything you cannot map becomes `UNKNOWN`.

```python
from vampire_survivors_env.features import FeatureProvider, GameState, ItemSlot, Mode, build_space

class MyProvider(FeatureProvider):
    def observation_space(self):
        return build_space()

    def read(self):
        if not self.game_is_in_a_run():
            return None                      # the env substitutes the null frame
        return GameState(
            mode=Mode.PLAYING, stage_time_s=self.clock(), hp=self.hp(), max_hp=self.max_hp(),
            level=self.level(), xp_frac=self.xp_frac(),
            weapons=[ItemSlot(index, level) for index, level in self.weapons()],
        ).encode()
```

Return `None` outside a run (menus, loading screens): the env fills in the null frame. To check a provider against the spec:

```bash
pytest tests/conformance --provider "my_package.my_module:MyProvider" \
    --provider-kwargs '{"some": "arg"}' [--provider-pid PID] [--read-budget-ms 1.0]
```

`vampire_survivors_env.features.mock.MockProvider` replays synthetic or recorded traces (`save_trace` / `load_trace`). Use it to develop consumers without the game.
