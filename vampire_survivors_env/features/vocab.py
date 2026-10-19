"""Stable item vocabulary for the feature spec (backed by vocab.json).

Indices are part of the spec: the file is append-only, index 0 is EMPTY and
index 1 is UNKNOWN.  Providers map whatever the game uses internally onto
vocabulary keys (e.g. "WHIP"), so game enum values never reach the policy.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from vampire_survivors_env.features.spec import EMPTY_INDEX, UNKNOWN_INDEX, VOCAB_CAPACITY

VOCAB_PATH = Path(__file__).with_name("vocab.json")

CATEGORIES = frozenset(
    {"reserved", "weapon", "evolution", "union", "gift", "counterpart", "special_weapon", "passive", "pickup"}
)


@dataclass(frozen=True)
class VocabEntry:
    index: int
    key: str
    name: str
    category: str
    dlc: Optional[str] = None
    deprecated: bool = False


class Vocab:
    """Bidirectional key <-> index lookup over vocab.json entries."""

    def __init__(self, entries: list[VocabEntry]) -> None:
        self.entries = entries
        self._by_key = {e.key: e for e in entries}

    def __len__(self) -> int:
        return len(self.entries)

    def index(self, key: Optional[str]) -> int:
        """Vocab index for key; EMPTY for None, UNKNOWN for unrecognised keys."""
        if key is None:
            return EMPTY_INDEX
        entry = self._by_key.get(key)
        return entry.index if entry is not None else UNKNOWN_INDEX

    def key(self, index: int) -> str:
        return self.entries[index].key

    def indices(self, *categories: str) -> list[int]:
        """All non-deprecated indices in the given categories."""
        return [e.index for e in self.entries if e.category in categories and not e.deprecated]


def validate_vocab_doc(doc: dict) -> list[VocabEntry]:
    """Parse and check a vocab.json document; raise ValueError on any problem."""
    entries = [VocabEntry(**e) for e in doc["entries"]]
    for pos, e in enumerate(entries):
        if e.index != pos:
            raise ValueError(f"vocab entry {e.key!r} has index {e.index}, expected {pos} (indices must be contiguous)")
        if e.category not in CATEGORIES:
            raise ValueError(f"vocab entry {e.key!r} has unknown category {e.category!r}")
    keys = [e.key for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("vocab keys must be unique")
    if keys[EMPTY_INDEX] != "EMPTY" or keys[UNKNOWN_INDEX] != "UNKNOWN":
        raise ValueError("vocab indices 0 and 1 must be EMPTY and UNKNOWN")
    if len(entries) > VOCAB_CAPACITY:
        raise ValueError(f"vocab has {len(entries)} entries; spec capacity is {VOCAB_CAPACITY}")
    return entries


@lru_cache(maxsize=1)
def load_vocab(path: Path = VOCAB_PATH) -> Vocab:
    """Load and validate the vocabulary (cached)."""
    with open(path, encoding="utf-8") as f:
        return Vocab(validate_vocab_doc(json.load(f)))
