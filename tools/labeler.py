"""Interactive labeler for raw phase-detection screenshots.

Displays each unlabeled screenshot fullscreen and records a keypress label.
Labels are appended to data/labels.csv as you go — quitting mid-session is safe.

Keys:
  p  playing
  l  levelup
  c  card_selection
  g  gameover
  r  results
  s  skip (leave unlabeled for now)
  u  undo last label written this session
  q  quit
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "screenshots" / "raw"
LABELS_CSV = ROOT / "data" / "labels.csv"

PHASE_KEYS: dict[int, str] = {
    ord("p"): "playing",
    ord("l"): "levelup",
    ord("c"): "card_selection",
    ord("g"): "gameover",
    ord("r"): "results",
}

PHASE_COLORS: dict[str, tuple[int, int, int]] = {
    "playing":        (0, 200, 0),
    "levelup":        (0, 200, 200),
    "card_selection": (200, 200, 0),
    "gameover":       (0, 0, 200),
    "results":        (200, 140, 0),
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_labeled() -> set[str]:
    if not LABELS_CSV.exists():
        return set()
    with open(LABELS_CSV, newline="") as f:
        return {row["filename"] for row in csv.DictReader(f) if row.get("filename")}


def _append(filename: str, phase: str, level: str) -> None:
    write_header = not LABELS_CSV.exists() or LABELS_CSV.stat().st_size == 0
    with open(LABELS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "phase", "level"])
        if write_header:
            w.writeheader()
        w.writerow({"filename": filename, "phase": phase, "level": level})


def _remove(filename: str) -> None:
    if not LABELS_CSV.exists():
        return
    rows: list[dict] = []
    with open(LABELS_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("filename") != filename]
    with open(LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "phase", "level"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Filename → level name
# Filename format: YYYYMMDD_HHMMSS_microseconds_level_slug.png
# ---------------------------------------------------------------------------

def _level_from_filename(name: str) -> str:
    parts = name.removesuffix(".png").split("_")
    # First 3 parts: date, time, microseconds
    return " ".join(parts[3:]).replace("_", " ").title() if len(parts) > 3 else "unknown"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

BAR_H = 50


def _draw_ui(img_bgr: np.ndarray, name: str, idx: int, total: int, phase: str | None) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    canvas = np.zeros((h + BAR_H, w, 3), dtype=np.uint8)
    canvas[BAR_H:] = img_bgr

    # Info bar
    cv2.rectangle(canvas, (0, 0), (w, BAR_H), (25, 25, 25), -1)
    cv2.putText(canvas, f"[{idx}/{total}]  {name}",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "p=playing  l=levelup  c=card_sel  g=gameover  r=results    s=skip  u=undo  q=quit",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    if phase is not None:
        color = PHASE_COLORS.get(phase, (255, 255, 255))
        label_text = f"  {phase}  "
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(canvas, (w - tw - 20, 6), (w - 4, 6 + th + 8), (40, 40, 40), -1)
        cv2.putText(canvas, label_text, (w - tw - 16, 6 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    labeled = _load_labeled()
    all_images = sorted(RAW_DIR.glob("*.png"))
    queue = [p for p in all_images if p.name not in labeled]

    if not queue:
        print("Nothing to label — run screenshot_collector.py first (or everything is already labeled).")
        return

    print(f"{len(labeled)} already labeled. {len(queue)} remaining.")

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeler", 960, 640)

    history: list[str] = []  # filenames labeled this session, for undo
    i = 0

    while i < len(queue):
        path = queue[i]
        img = cv2.imread(str(path))
        if img is None:
            i += 1
            continue

        cv2.imshow("Labeler", _draw_ui(img, path.name, i + 1, len(queue), None))

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in PHASE_KEYS:
                phase = PHASE_KEYS[key]
                level = _level_from_filename(path.name)
                _append(path.name, phase, level)
                history.append(path.name)
                # Flash the label briefly before moving on
                cv2.imshow("Labeler", _draw_ui(img, path.name, i + 1, len(queue), phase))
                cv2.waitKey(350)
                i += 1
                break

            elif key == ord("s"):
                i += 1
                break

            elif key == ord("u"):
                if history:
                    last = history.pop()
                    _remove(last)
                    print(f"  Undid label for {last}")
                    # Step back to re-show that image if it's in queue
                    try:
                        i = queue.index(RAW_DIR / last)
                    except ValueError:
                        i = max(0, i - 1)
                else:
                    print("  Nothing to undo.")
                break

            elif key in (ord("q"), 27):  # q or Esc
                print("Quit.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"Done. Labels at {LABELS_CSV}")


if __name__ == "__main__":
    main()
