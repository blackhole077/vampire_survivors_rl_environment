# ---------------------------------------------------------------------------
# Phase detection
#
# Coordinates are into the raw game-window frame (no window chrome),
# rows=y, cols=x, calibrated at 640×480.
#
# Detection cascade (see _detect_game_phase in env.py):
#   1. Red-channel dominance → gameover / results
#   2. Center-panel brightness → results vs gameover
#   3. Edge density in card region → levelup / card_selection
#   4. Default → playing
#
# _RED_RATIO_THRESHOLD, _RESULTS_PANEL_THRESHOLD, _MENU_EDGE_THRESHOLD are
# calibrated by tools/threshold_tuner.py from the labeled screenshot dataset.
# Run that tool and paste its output here whenever the dataset changes.
_LEVELUP_ROW_SLICE = slice(70, 170)  # card-grid region for edge-density check
_LEVELUP_COL_SLICE = slice(250, 400)
_LEVELUP_CONSEC_FRAMES = 3  # consecutive menu frames before committing
_LEVELUP_BRIGHTNESS_THRESHOLD = 110  # midpoint: levelup~102, playing~167

# Thresholds — replace with values from tools/threshold_tuner.py once data is collected
_RED_RATIO_THRESHOLD = 1.6  # min(R/G, R/B) above this → gameover or results
_RESULTS_PANEL_THRESHOLD = (
    100.0  # center-region mean above this → results (not gameover)
)
_MENU_EDGE_THRESHOLD = 500.0  # Laplacian variance above this → menu screen

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
