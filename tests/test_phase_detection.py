"""Regression tests for phase detection using the labeled screenshot dataset.

Each labeled screenshot becomes one test case. Tests will be skipped if the
labels CSV is empty or missing (e.g. on a clean checkout before data collection).

Run:
    pytest tests/test_phase_detection.py -v

After changing constants.py thresholds, run this suite to verify no regressions.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS_CSV = ROOT / "data" / "labels.csv"
RAW_DIR = ROOT / "data" / "screenshots" / "raw"

# gameplay_started assumption per phase (matches labeler convention)
_GAMEPLAY_STARTED: dict[str, bool] = {
    "playing": True,
    "levelup": True,
    "gameover": True,
    "results": True,
    "card_selection": False,
}


# ---------------------------------------------------------------------------
# Standalone detector — mirrors _detect_game_phase without env instance state
# ---------------------------------------------------------------------------

def _detect(frame: np.ndarray, gameplay_started: bool) -> str:
    from vampire_survivors_env.constants import (
        _CARD_RED_RATIO_THRESHOLD,
        _LEVELUP_CENTER_MEAN_THRESHOLD,
        _LEVELUP_COL_SLICE,
        _LEVELUP_ROW_SLICE,
        _MENU_CENTER_THRESHOLD,
        _MENU_EDGE_THRESHOLD,
        _RED_RATIO_THRESHOLD,
        _RESULTS_PANEL_THRESHOLD,
    )

    h, w = frame.shape[:2]
    eps = 1e-6

    center = frame[h // 4 : 3 * h // 4, w // 6 : 5 * w // 6]
    center_mean = float(center.mean())

    r_mean = float(frame[:, :, 0].mean())
    g_mean = float(frame[:, :, 1].mean())
    b_mean = float(frame[:, :, 2].mean())
    r_ratio = min(r_mean / (g_mean + eps), r_mean / (b_mean + eps))

    # Step 1 — bright overlay → levelup
    if center_mean > _LEVELUP_CENTER_MEAN_THRESHOLD:
        return "levelup"

    # Step 2 — high edge density in card region → menu overlay.
    # Requires either high center brightness (card_sel A ~88, levelup ~95) or
    # very high r_ratio (red-tinted card_sel B ~2.9) to avoid false positives
    # on dark-level playing/gameover frames (center ~40-47, r_ratio ~0.60-1.58).
    region = frame[_LEVELUP_ROW_SLICE, _LEVELUP_COL_SLICE]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    edge_density = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if edge_density > _MENU_EDGE_THRESHOLD and (
        center_mean > _MENU_CENTER_THRESHOLD or r_ratio > _CARD_RED_RATIO_THRESHOLD
    ):
        return "card_selection" if not gameplay_started else "levelup"

    # Step 3 — red-tinted screen → gameover or results
    if r_ratio > _RED_RATIO_THRESHOLD:
        return "results" if center_mean > _RESULTS_PANEL_THRESHOLD else "gameover"

    return "playing"


# ---------------------------------------------------------------------------
# Parametrize from labels.csv
# ---------------------------------------------------------------------------

def _load_cases() -> list[tuple[str, str, str]]:
    if not LABELS_CSV.exists():
        return []
    with open(LABELS_CSV, newline="") as f:
        return [
            (row["filename"], row["phase"], row.get("level", ""))
            for row in csv.DictReader(f)
            if row.get("filename") and row.get("phase")
        ]


CASES = _load_cases()


@pytest.mark.skipif(not CASES, reason="No labeled data — run tools/labeler.py first")
@pytest.mark.parametrize("filename,expected,level", CASES, ids=[c[0] for c in CASES])
def test_phase_detection(filename: str, expected: str, level: str) -> None:
    path = RAW_DIR / filename
    assert path.exists(), f"Screenshot missing from raw dir: {path}"

    img_bgr = cv2.imread(str(path))
    assert img_bgr is not None, f"cv2 could not load: {path}"

    frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gameplay_started = _GAMEPLAY_STARTED.get(expected, True)
    detected = _detect(frame, gameplay_started)

    assert detected == expected, (
        f"{filename} [{level}]: expected={expected!r}  got={detected!r}"
    )
