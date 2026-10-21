"""Evolution recipes (evolutions.json) and which picks move an evolution closer."""

import pytest

from vampire_survivors_env.features.evolutions import load_recipes, validate_recipes_doc
from vampire_survivors_env.features.vocab import load_vocab


@pytest.fixture(scope="module")
def recipes():
    return load_recipes()


def test_recipes_load_and_name_only_vocab_items(recipes):
    assert len(recipes.recipes) > 100
    whip = [r for r in recipes.using("WHIP") if r.result == "BLOODY_TEAR"]
    assert whip and whip[0].ingredients == ("WHIP", "HOLLOW_HEART") and not whip[0].max_level_passives


def test_unknown_item_is_rejected():
    doc = {"recipes": [{"result": "BLOODY_TEAR", "kind": "evolution", "ingredients": ["WHIP", "NOT_AN_ITEM"],
                        "max_level_passives": False}]}
    with pytest.raises(ValueError, match="NOT_AN_ITEM"):
        validate_recipes_doc(doc, load_vocab())


@pytest.mark.parametrize("item, new, owned, expected", [
    ("HOLLOW_HEART", True, {"WHIP", "HOLLOW_HEART"}, True),  # the passive Whip needs
    ("WHIP", True, {"WHIP", "HOLLOW_HEART"}, True),  # the weapon Hollow Heart needs
    ("WHIP", False, {"WHIP", "HOLLOW_HEART"}, True),  # Whip evolves at max level
    ("WHIP", False, {"WHIP"}, False),  # no Hollow Heart yet
    ("HOLLOW_HEART", True, {"HOLLOW_HEART", "KNIFE"}, False),  # no Whip
    ("HOLLOW_HEART", False, {"WHIP", "HOLLOW_HEART"}, False),  # any Hollow Heart level will do
    ("HOLLOW_HEART", True, {"BLOODY_TEAR", "HOLLOW_HEART"}, False),  # already evolved, Whip gone
    ("SILVER_RING", True, {"CLOCK_LANCET", "SILVER_RING"}, False),  # Gold Ring still missing
    ("SILVER_RING", True, {"CLOCK_LANCET", "SILVER_RING", "GOLD_RING"}, True),
    ("SILVER_RING", False, {"CLOCK_LANCET", "SILVER_RING", "GOLD_RING"}, True),  # rings must be maxed too
    ("BRACELET", False, {"BRACELET"}, True),  # evolves alone
    ("PEACHONE", True, {"PEACHONE", "EBONY_WINGS"}, True),  # union
    ("SPINACH", True, {"SPINACH"}, False),
    ("UNKNOWN", True, {"UNKNOWN"}, False),
])
def test_contributes(recipes, item, new, owned, expected):
    assert recipes.contributes(item, new, owned) is expected
