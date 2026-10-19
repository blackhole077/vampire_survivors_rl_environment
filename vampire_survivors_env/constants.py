# ---------------------------------------------------------------------------
# Phase detection
#
# Coordinates are into the raw game-window frame (no window chrome),
# rows=y, cols=x, calibrated at 640×480.
#
# Detection cascade (see _detect_game_phase in env.py):
#   1. Center-panel brightness  > _LEVELUP_CENTER_MEAN_THRESHOLD → levelup
#   2. Edge density in card region > _MENU_EDGE_THRESHOLD        → card_selection / levelup
#   3. Red-channel dominance    > _RED_RATIO_THRESHOLD           → gameover / results
#   4. Default                                                    → playing
#
# Step 1 must come before step 2: levelup and card_selection both have high
# edge density, but levelup has a very bright overlay (center_mean ~99) that
# card_selection never reaches (max ~89).
#
# Step 2 must come before step 3: one card_selection variant has a strongly
# red-tinted background (r_ratio ~2.9) but its edge density (~4040) is still
# above the threshold, so it is correctly caught before the red check fires.
#
# Thresholds calibrated by tools/threshold_tuner.py from the labeled dataset.
# Run that tool and paste its output here whenever the dataset changes.

_LEVELUP_ROW_SLICE = slice(70, 170)  # card-grid region for edge-density check
_LEVELUP_COL_SLICE = slice(250, 400)
_LEVELUP_CONSEC_FRAMES = 3  # consecutive menu frames before committing (env use)

# Step 1 — center brightness (h/4:3h/4, w/6:5w/6)
# levelup: 98.9–99.1   others max: 90.9 (results)
_LEVELUP_CENTER_MEAN_THRESHOLD = 95.0

# Step 2 — Laplacian variance of the card-grid region.
# Edge density alone is not sufficient on dark levels where playing frames
# (dense enemies / explosions) and gameover screens also reach ~4500–7800.
# A true menu overlay must ALSO satisfy one of:
#   • center_mean > _MENU_CENTER_THRESHOLD  (card_sel A ~88-89, levelup ~95)
#   • r_ratio     > _CARD_RED_RATIO_THRESHOLD (card_sel B red-tinted variant ~2.7-2.9)
# Dark-level playing (center ~40-44, r_ratio ~0.60) and gameover (center ~47,
# r_ratio ~1.58) satisfy neither condition and fall through to step 3.
_MENU_EDGE_THRESHOLD = 4000.0
_MENU_CENTER_THRESHOLD = 70.0       # min center brightness for edge check to fire
_CARD_RED_RATIO_THRESHOLD = 2.0     # min r_ratio for red-tinted card_selection variant

# Step 3 — red-channel dominance: min(R/G, R/B)
# gameover: 1.561–1.583   results: 1.207   levelup max: 0.886   playing max: 0.652
_RED_RATIO_THRESHOLD = 1.1

# Step 3b — separate results from gameover by center brightness
# results: 90.9   gameover: 53.1–53.2
_RESULTS_PANEL_THRESHOLD = 70.0

# ---------------------------------------------------------------------------
# Results screen — survival-time OCR
#
# The "Survived: MM:SS" value is right-aligned in the stats panel.
# ROI calibrated at 640×480.  Processing pipeline:
#   1. Crop _SURVIVED_TIME_ROI from the raw frame.
#   2. Upscale 8× with INTER_CUBIC.
#   3. Grayscale → THRESH_BINARY_INV at _SURVIVED_THRESH (panel bg ~116 → black,
#      bright text → white) to produce black-text-on-white for Tesseract.
#   4. Add 20px white border padding.
#   5. pytesseract PSM 7 (single line), whitelist "0123456789:".
#
_SURVIVED_TIME_ROI = (slice(113, 131), slice(285, 336))  # (y_slice, x_slice)
_SURVIVED_THRESH = 130   # grayscale threshold separating text from panel bg
_SURVIVED_OCR_SCALE = 8  # upscale factor before OCR

_VS_WM_CLASS = ("VampireSurvivors.exe", "VampireSurvivors.exe")
_VS_WM_NAME = "Vampire Survivors"
_MOVE_KEYS = ["w", "a", "s", "d"]
_MENU_KEYS = ["w", "a", "s", "d", "space", "Escape"]
_ACTION_KEYS: list[frozenset[str]] = [
    frozenset(),  # 0: stay
    frozenset({"w"}),  # 1: up
    frozenset({"w", "d"}),  # 2: up-right
    frozenset({"d"}),  # 3: right
    frozenset({"s", "d"}),  # 4: down-right
    frozenset({"s"}),  # 5: down
    frozenset({"s", "a"}),  # 6: down-left
    frozenset({"a"}),  # 7: left
    frozenset({"w", "a"}),  # 8: up-left
    ### Menu-specific actions ###
    frozenset({"space"}),  # 9: Space (confirm)
    # frozenset({"Escape"}),  # 10: Escape (cancel)
]
