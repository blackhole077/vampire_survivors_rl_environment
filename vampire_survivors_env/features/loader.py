"""Resolve a feature provider from config without the core importing any implementation."""

import importlib
from importlib.metadata import entry_points
from typing import Any, Optional, Union

ENTRY_POINT_GROUP = "vampire_survivors_env.feature_providers"


def load_provider(spec: Union[str, object], kwargs: Optional[dict[str, Any]] = None) -> object:
    """Return a provider instance from an instance, an import path, or an entry-point name.

    spec may be:
      - an already-constructed provider (returned as-is; kwargs must be empty),
      - "package.module:Name" — Name is a class or factory called with **kwargs,
      - "name" (no colon) — an installed entry point in ENTRY_POINT_GROUP.
    """
    kwargs = kwargs or {}
    if not isinstance(spec, str):
        if kwargs:
            raise ValueError("feature_provider_kwargs only apply when feature_provider is a string")
        return spec

    if ":" in spec:
        module_name, _, attr = spec.partition(":")
        factory = getattr(importlib.import_module(module_name), attr)
    else:
        matches = entry_points(group=ENTRY_POINT_GROUP, name=spec)
        if not matches:
            raise LookupError(
                f"No feature provider entry point named {spec!r} in group {ENTRY_POINT_GROUP!r}. "
                f"Use 'package.module:ClassName' or install a package that registers it."
            )
        factory = next(iter(matches)).load()
    return factory(**kwargs)
