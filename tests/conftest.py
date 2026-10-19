"""Shared pytest options for the feature-provider conformance suite.

    pytest tests/conformance                                  # mock providers
    pytest tests/conformance --provider "pkg.mod:Class" \\
        --provider-kwargs '{"offsets": "..."}' --provider-pid 12345
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRACE_DIR = ROOT / "tests" / "data" / "feature_traces"

# (id, import path, kwargs) — the default providers the conformance suite runs against.
DEFAULT_PROVIDERS = [
    ("mock-synthetic", "vampire_survivors_env.features.mock:MockProvider", {}),
    ("mock-enemies", "vampire_survivors_env.features.mock:MockProvider", {"optional_keys": ["enemies"], "seed": 1}),
    ("mock-trace-file", "vampire_survivors_env.features.mock:MockProvider", {"trace": str(TRACE_DIR / "mock_run.json")}),
]


def pytest_addoption(parser):
    group = parser.getgroup("feature providers")
    group.addoption("--provider", action="append", default=[], help="Provider import path or entry-point name; repeatable.")
    group.addoption("--provider-kwargs", default="{}", help="JSON kwargs for every --provider.")
    group.addoption("--provider-pid", type=int, default=None, help="pid passed to provider.attach().")
    group.addoption("--conformance-reads", type=int, default=500, help="read() calls per provider.")
    group.addoption("--read-budget-ms", type=float, default=1.0, help="Max mean read() time in ms.")


def pytest_generate_tests(metafunc):
    if "provider_config" not in metafunc.fixturenames:
        return
    chosen = metafunc.config.getoption("--provider")
    if chosen:
        kwargs = json.loads(metafunc.config.getoption("--provider-kwargs"))
        configs = [(path, path, kwargs) for path in chosen]
    else:
        configs = DEFAULT_PROVIDERS
    metafunc.parametrize("provider_config", [c[1:] for c in configs], ids=[c[0] for c in configs])
