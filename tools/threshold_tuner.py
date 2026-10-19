"""Analyse the labeled screenshot dataset and recommend phase-detection thresholds.

Extracts the same features that _detect_game_phase will use, prints per-phase
distribution statistics, suggests midpoint thresholds, and shows a confusion
matrix so you can validate before hard-coding values into constants.py.

Usage:
    python tools/threshold_tuner.py

Optional — override thresholds manually to test a candidate:
    python tools/threshold_tuner.py --red-ratio 1.55 --panel 105 --edge 420
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vampire_survivors_env.constants import _LEVELUP_COL_SLICE, _LEVELUP_ROW_SLICE

LABELS_CSV = ROOT / "data" / "labels.csv"
RAW_DIR = ROOT / "data" / "screenshots" / "raw"

PHASES = ["playing", "levelup", "card_selection", "gameover", "results"]

# gameplay_started assumption per phase label (card_selection happens before gameplay)
_GAMEPLAY_STARTED: dict[str, bool] = {
    "playing": True,
    "levelup": True,
    "gameover": True,
    "results": True,
    "card_selection": False,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(frame: np.ndarray) -> dict[str, float]:
    """Return the scalar features used by the detection cascade."""
    h, w = frame.shape[:2]

    r = frame[:, :, 0].astype(np.float64)
    g = frame[:, :, 1].astype(np.float64)
    b = frame[:, :, 2].astype(np.float64)

    r_mean = float(r.mean())
    g_mean = float(g.mean())
    b_mean = float(b.mean())

    eps = 1e-6
    # Use the smaller of the two ratios so a single channel spike doesn't dominate
    r_ratio = min(r_mean / (g_mean + eps), r_mean / (b_mean + eps))

    # Center panel brightness: results has a large gray panel, gameover does not
    center = frame[h // 4 : 3 * h // 4, w // 6 : 5 * w // 6]
    center_brightness = float(center.mean())

    # Edge density in the card-region: menu screens have high-contrast card borders
    region = frame[_LEVELUP_ROW_SLICE, _LEVELUP_COL_SLICE]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    edge_density = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "overall_mean": float(frame.mean()),
        "r_ratio": r_ratio,
        "center_brightness": center_brightness,
        "edge_density": edge_density,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset() -> list[dict]:
    if not LABELS_CSV.exists():
        print(f"No labels found at {LABELS_CSV}. Run labeler.py first.")
        sys.exit(1)

    rows: list[dict] = []
    with open(LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            path = RAW_DIR / row["filename"]
            if not path.exists():
                print(f"  Warning: {row['filename']} not found, skipping.")
                continue
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                continue
            frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            feats = extract_features(frame)
            rows.append({"phase": row["phase"], "level": row.get("level", ""), **feats})
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(rows: list[dict]) -> None:
    features = ["r_ratio", "center_brightness", "edge_density",
                "r_mean", "g_mean", "b_mean", "overall_mean"]

    by_phase: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_phase[r["phase"]].append(r)

    print("\n=== Feature Distributions by Phase ===\n")
    col_w = 16
    for feat in features:
        print(f"  {feat}:")
        for phase in PHASES:
            vals = [r[feat] for r in by_phase.get(phase, [])]
            if not vals:
                continue
            arr = np.array(vals)
            print(
                f"    {phase:{col_w}s}  n={len(vals):3d}"
                f"  mean={arr.mean():8.2f}"
                f"  std={arr.std():7.2f}"
                f"  min={arr.min():8.2f}"
                f"  max={arr.max():8.2f}"
            )
        print()


# ---------------------------------------------------------------------------
# Threshold suggestion (midpoint between class means)
# ---------------------------------------------------------------------------

def suggest_thresholds(rows: list[dict]) -> dict[str, float]:
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_phase[r["phase"]].append(r)

    def midpoint(phases_a: list[str], phases_b: list[str], feat: str) -> float | None:
        a = [r[feat] for p in phases_a for r in by_phase.get(p, [])]
        b = [r[feat] for p in phases_b for r in by_phase.get(p, [])]
        if not a or not b:
            return None
        return float((np.mean(a) + np.mean(b)) / 2)

    thresholds: dict[str, float] = {}

    # r_ratio: gameover+results are red-dominant vs everything else
    t = midpoint(["gameover", "results"], ["playing", "levelup", "card_selection"], "r_ratio")
    if t is not None:
        thresholds["_RED_RATIO_THRESHOLD"] = round(t, 3)

    # center_brightness: results has a bright gray panel, gameover does not
    t = midpoint(["results"], ["gameover"], "center_brightness")
    if t is not None:
        thresholds["_RESULTS_PANEL_THRESHOLD"] = round(t, 2)

    # edge_density: menu screens (levelup + card_selection) vs playing
    t = midpoint(["levelup", "card_selection"], ["playing"], "edge_density")
    if t is not None:
        thresholds["_MENU_EDGE_THRESHOLD"] = round(t, 1)

    return thresholds


# ---------------------------------------------------------------------------
# Detection (mirrors _detect_game_phase logic without env state)
# ---------------------------------------------------------------------------

def _detect(feats: dict, thresholds: dict, gameplay_started: bool) -> str:
    if feats["r_ratio"] > thresholds["_RED_RATIO_THRESHOLD"]:
        if feats["center_brightness"] > thresholds["_RESULTS_PANEL_THRESHOLD"]:
            return "results"
        return "gameover"
    if feats["edge_density"] > thresholds["_MENU_EDGE_THRESHOLD"]:
        return "card_selection" if not gameplay_started else "levelup"
    return "playing"


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def print_confusion(rows: list[dict], thresholds: dict) -> None:
    col_w = 14
    header = f"{'actual \\ pred':{col_w}s}" + "".join(f"{p:{col_w}s}" for p in PHASES)
    print(f"\n=== Confusion Matrix ===\n{header}\n{'-' * len(header)}")

    matrix: dict[str, dict[str, int]] = {p: {q: 0 for q in PHASES} for p in PHASES}
    correct = 0
    for row in rows:
        actual = row["phase"]
        gs = _GAMEPLAY_STARTED.get(actual, True)
        predicted = _detect(row, thresholds, gs)
        matrix[actual][predicted] += 1
        if actual == predicted:
            correct += 1

    for actual in PHASES:
        if not any(matrix[actual].values()):
            continue
        row_str = f"{actual:{col_w}s}" + "".join(f"{matrix[actual][p]:{col_w}d}" for p in PHASES)
        print(row_str)

    acc = correct / len(rows) if rows else 0.0
    print(f"\nAccuracy: {correct}/{len(rows)} = {acc:.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tune phase-detection thresholds.")
    parser.add_argument("--red-ratio", type=float, help="Override _RED_RATIO_THRESHOLD")
    parser.add_argument("--panel", type=float, help="Override _RESULTS_PANEL_THRESHOLD")
    parser.add_argument("--edge", type=float, help="Override _MENU_EDGE_THRESHOLD")
    args = parser.parse_args()

    print(f"Loading dataset from {LABELS_CSV} ...")
    rows = load_dataset()
    if not rows:
        print("No data loaded. Exiting.")
        return

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["phase"]] += 1
    print(f"Loaded {len(rows)} labeled screenshots:")
    for phase in PHASES:
        n = counts.get(phase, 0)
        if n:
            print(f"  {phase:16s}: {n}")

    print_stats(rows)

    suggested = suggest_thresholds(rows)

    # Allow CLI overrides
    thresholds = {
        "_RED_RATIO_THRESHOLD":    args.red_ratio if args.red_ratio is not None else suggested.get("_RED_RATIO_THRESHOLD", 1.6),
        "_RESULTS_PANEL_THRESHOLD": args.panel    if args.panel    is not None else suggested.get("_RESULTS_PANEL_THRESHOLD", 100.0),
        "_MENU_EDGE_THRESHOLD":    args.edge      if args.edge     is not None else suggested.get("_MENU_EDGE_THRESHOLD", 500.0),
    }

    print("=== Thresholds in use ===")
    for k, v in thresholds.items():
        src = "(CLI override)" if getattr(args, k.lstrip("_").lower().replace("threshold", "").replace("ratio", "ratio").replace("panel", "panel").replace("menu_edge", "edge"), None) is not None else "(suggested)"
        print(f"  {k} = {v}  {src}")

    print_confusion(rows, thresholds)

    print("\n--- Paste into vampire_survivors_env/constants.py ---")
    for k, v in thresholds.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()
