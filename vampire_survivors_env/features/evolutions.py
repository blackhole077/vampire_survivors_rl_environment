"""Evolution, union and gift recipes (backed by evolutions.json, built by tools/build_evolutions.py).

An evolution needs every ingredient owned when a qualifying chest is opened:
weapons at max level, and passives at max level too for recipes marked
max_level_passives (otherwise any level will do).  Item names are vocab.json
keys, as in vocab.py.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from vampire_survivors_env.features.vocab import Vocab, load_vocab

EVOLUTIONS_PATH = Path(__file__).with_name("evolutions.json")


@dataclass(frozen=True)
class Recipe:
    result: str
    kind: str  # "evolution", "union" or "gift"
    ingredients: tuple[str, ...]
    max_level_passives: bool
    dlc: Optional[str] = None


class Recipes:
    def __init__(self, recipes: list[Recipe], vocab: Vocab) -> None:
        self.recipes = recipes
        self._passives = {vocab.key(i) for i in vocab.indices("passive")}
        self._using: dict[str, list[Recipe]] = {}
        for r in recipes:
            for item in r.ingredients:
                self._using.setdefault(item, []).append(r)

    def using(self, item: str) -> list[Recipe]:
        """The recipes item is an ingredient of."""
        return self._using.get(item, [])

    def is_passive(self, item: str) -> bool:
        return item in self._passives

    def contributes(self, item: str, new: bool, owned: Iterable[str]) -> bool:
        """Whether gaining item (new: first copy; else one more level) moves an evolution closer.

        owned is everything held after the pick.  It does when item is an ingredient of a
        recipe whose other ingredients are held already, or that has no other ingredients
        (Bracelet), and the level matters: a new item always, a weapon level always (weapons
        evolve at max level), a passive level only for max_level_passives recipes.  An
        ingredient whose partners aren't held yet doesn't count, nor does a recipe whose
        result is held already.
        """
        owned = set(owned)
        for r in self.using(item):
            if r.result in owned:
                continue
            if not new and self.is_passive(item) and not r.max_level_passives:
                continue
            others = [i for i in r.ingredients if i != item]
            if all(i in owned for i in others):
                return True
        return False


def validate_recipes_doc(doc: dict, vocab: Vocab) -> list[Recipe]:
    """Parse and check an evolutions.json document; raise ValueError on any problem."""
    recipes = []
    for r in doc["recipes"]:
        recipe = Recipe(r["result"], r["kind"], tuple(r["ingredients"]), r["max_level_passives"], r.get("dlc"))
        for key in (recipe.result, *recipe.ingredients):
            if vocab.index(key) == vocab.index("UNKNOWN"):
                raise ValueError(f"recipe for {recipe.result!r} names {key!r}, which is not in vocab.json")
        if not recipe.ingredients:
            raise ValueError(f"recipe for {recipe.result!r} has no ingredients")
        recipes.append(recipe)
    return recipes


@lru_cache(maxsize=1)
def load_recipes(path: Path = EVOLUTIONS_PATH) -> Recipes:
    """Load and validate the recipes (cached)."""
    vocab = load_vocab()
    with open(path, encoding="utf-8") as f:
        return Recipes(validate_recipes_doc(json.load(f), vocab), vocab)
