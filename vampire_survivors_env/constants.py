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

# Step 2 — Laplacian variance of the card-grid region
# card_selection: 4039–6126   playing max: 3672
_MENU_EDGE_THRESHOLD = 4000.0

# Step 3 — red-channel dominance: min(R/G, R/B)
# gameover: 1.561   results: 1.207   levelup max: 0.886   playing max: 0.641
_RED_RATIO_THRESHOLD = 1.1

# Step 3b — separate results from gameover by center brightness
# results: 90.9   gameover: 53.1–53.2
_RESULTS_PANEL_THRESHOLD = 70.0

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
