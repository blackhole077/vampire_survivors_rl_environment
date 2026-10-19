"""Capture raw game-window screenshots for the phase-detection dataset.

Usage — manual (press Enter in terminal to capture):
    python tools/screenshot_collector.py --label playing

Usage — automatic (capture every N seconds):
    python tools/screenshot_collector.py --label levelup --interval 2

Usage — automatic with a cap:
    python tools/screenshot_collector.py --label gameover --interval 1 --count 50

The --label argument becomes the phase slug embedded in every filename and is
also what the labeler will show as the "level" column.  Use the same names the
phase detector recognises: playing, levelup, card_selection, gameover, results.

Saved to: data/screenshots/raw/{timestamp}_{label}.png
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np
from ewmhlib import EwmhRoot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vampire_survivors_env.utils.window_utils import (
    GameWindow,
    capture_game_window,
    find_game_window,
)

OUT_DIR = ROOT / "data" / "screenshots" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_LABELS = {"playing", "levelup", "card_selection", "gameover", "results"}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _save(frame: np.ndarray, label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = OUT_DIR / f"{ts}_{_slug(label)}.png"
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path


def _show_preview(sct: mss.base.MSSBase, window: GameWindow) -> None:
    """Capture once and show a preview window so the user can verify alignment."""
    frame = capture_game_window(sct, window)
    print(f"  Window : {window.width}x{window.height}  at ({window.x}, {window.y})  xid={window.xid}")
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (640, 360))
    cv2.imshow("Preview — press any key to start capturing", small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect phase-detection screenshots.")
    parser.add_argument(
        "--label",
        required=True,
        help=(
            f"Phase label embedded in filenames. Known phases: {', '.join(sorted(VALID_LABELS))}. "
            "Any string is accepted; unknown labels print a warning."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Auto-capture interval in seconds. 0 (default) = manual mode (press Enter).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Stop after this many captures (auto mode only). 0 = no limit.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the alignment preview window and start capturing immediately.",
    )
    args = parser.parse_args()

    if args.label not in VALID_LABELS:
        print(f"Warning: '{args.label}' is not a recognised phase label. Known: {sorted(VALID_LABELS)}")

    ewmh_root = EwmhRoot()
    window = find_game_window(ewmh_root)
    sct = mss.mss()

    if not args.no_preview:
        _show_preview(sct, window)

    print(f"Output : {OUT_DIR}")
    print(f"Label  : {args.label!r}")

    count = 0
    try:
        if args.interval > 0:
            limit = args.count or None
            print(
                f"Auto-capturing every {args.interval}s"
                + (f" (max {limit})" if limit else "")
                + ". Ctrl+C to stop.\n"
            )
            while limit is None or count < limit:
                frame = capture_game_window(sct, window)
                path = _save(frame, args.label)
                count += 1
                print(f"  [{count:4d}] {path.name}")
                time.sleep(args.interval)
        else:
            print("Press Enter to capture. 'q' + Enter to quit.\n")
            while True:
                try:
                    key = input()
                except EOFError:
                    break
                if key.strip().lower() == "q":
                    break
                frame = capture_game_window(sct, window)
                path = _save(frame, args.label)
                count += 1
                print(f"  [{count:4d}] {path.name}")
    except KeyboardInterrupt:
        pass
    finally:
        sct.close()
        print(f"\nDone. {count} frame(s) saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
