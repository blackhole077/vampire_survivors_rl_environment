"""
Sanity-check script for _find_vs_xid and _get_abs_geometry.
Run from the repo root with the game already open:

    python scripts/test_find_window.py
"""

from ewmhlib import EwmhRoot, EwmhWindow
from utils.window_utils import (
    GameWindow,
    find_game_window,
    find_vs_xid,
    get_abs_geometry,
)
import Xlib.X
import Xlib.XK
import Xlib.ext.xtest

ewmh_root = EwmhRoot()

# ── 1. Find the XID ────────────────────────────────────────────────────────────
print("Searching for Vampire Survivors window...")
xid = find_vs_xid(ewmh_root.root)

if xid is None:
    print("NOT FOUND — is the game running?")
    raise SystemExit(1)

print(f"  xid      : {xid}")

# ── 2. Cross-check name and class ─────────────────────────────────────────────
win = EwmhWindow(xid)
print(f"  getName()    : {win.getName()!r}")
print(f"  get_wm_class : {win.xWindow.get_wm_class()}")
print(f"  getPid()     : {win.getPid()}")

# ── 3. Resolve absolute geometry ──────────────────────────────────────────────
x, y, w, h = get_abs_geometry(win.xWindow, ewmh_root.root.id)
print(f"  geometry : x={x} y={y} w={w} h={h}")

# ── 4. Build GameWindow ────────────────────────────────────────────────────────
gw = GameWindow(xid=xid, pid=win.getPid() or -1, x=x, y=y, width=w, height=h)
print(f"\nGameWindow: {gw}")

test_gw = find_game_window(ewmh_root=EwmhRoot())
print(test_gw)
print("-----------------")
print(gw)
assert test_gw == gw
