"""Optional game-state feature channel.  See spec.py for the contract."""

from vampire_survivors_env.features.evolutions import Recipe, Recipes, load_recipes
from vampire_survivors_env.features.spec import (
    SPEC_VERSION,
    FeatureFrame,
    FeatureProvider,
    GameState,
    IncompatibleProviderError,
    ItemSlot,
    Mode,
    build_space,
    check_compatible,
    core_space,
    encode_enemy_grid,
    encode_scalars,
    encode_slots,
    frame_problems,
    null_frame,
)
from vampire_survivors_env.features.vocab import Vocab, load_vocab

__all__ = [
    "SPEC_VERSION",
    "FeatureFrame",
    "FeatureProvider",
    "GameState",
    "IncompatibleProviderError",
    "ItemSlot",
    "Mode",
    "Recipe",
    "Recipes",
    "Vocab",
    "build_space",
    "check_compatible",
    "core_space",
    "encode_enemy_grid",
    "encode_scalars",
    "encode_slots",
    "frame_problems",
    "load_recipes",
    "load_vocab",
    "null_frame",
]
