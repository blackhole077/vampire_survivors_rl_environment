from functools import lru_cache
import time
from dataclasses import dataclass
from typing import Optional

import mss
import mss.base
import numpy as np
import Xlib.X
import Xlib.XK
from ewmhlib import EwmhRoot, EwmhWindow
from Xlib.X import KeyRelease, KeyPress
from Xlib.ext.xtest import fake_input
from Xlib.xobject.drawable import Window as XWindow
from Xlib.display import Display
from vampire_survivors_env.constants import _MENU_KEYS, _VS_WM_CLASS, _VS_WM_NAME


@dataclass
class GameWindow:
    """X11 window handle plus geometry needed for screen capture cropping."""

    xid: int  # X11 window ID
    pid: int  # OS process ID (for liveness checks)
    x: int  # absolute screen x (for mss crop region)
    y: int  # absolute screen y
    width: int
    height: int


def capture_game_window(sct: mss.base.MSSBase, window: GameWindow) -> np.ndarray:
    """Grab the game window and return an HxWx3 uint8 RGB array.

    mss returns BGRA; the alpha channel is dropped and channels are flipped to RGB.
    """
    region = {
        "top": window.y,
        "left": window.x,
        "width": window.width,
        "height": window.height,
    }
    raw = np.array(sct.grab(region), dtype=np.uint8)
    return raw[:, :, 2::-1]  # BGRA -> RGB, shape (H, W, 3)


def generate_keycodes(
    display: Display,
) -> dict[str, int]:
    return {
        k: display.keysym_to_keycode(Xlib.XK.string_to_keysym(k)) for k in _MENU_KEYS
    }


def focus_window(display: Display, window_xid: int) -> None:
    """Set X11 input focus to the game window so XTEST events are accepted by Wine/Proton.

    Raises the top-level WM-managed ancestor first so the WM does not override
    the subsequent set_input_focus call (common failure mode on Wine/Proton).
    """
    xwin = display.create_resource_object("window", window_xid)
    root = display.screen().root
    # Walk up to the direct child of root (the WM-managed top-level window).
    ancestor = xwin
    while True:
        parent = ancestor.query_tree().parent
        if parent.id == root.id:
            break
        ancestor = parent
    ancestor.raise_window()
    display.set_input_focus(xwin, Xlib.X.RevertToParent, Xlib.X.CurrentTime)
    display.sync()


def tap_key(display: Display, keycode: int, hold_secs: float = 0.05) -> None:
    """Press and release a single key via XTEST, blocking for hold_secs between events."""
    fake_input(display, KeyPress, keycode)
    display.flush()
    time.sleep(hold_secs)
    fake_input(display, KeyRelease, keycode)
    display.flush()


def find_vs_xid(win: XWindow) -> Optional[int]:
    """Walk the full X11 window tree and return the XID of the Vampire
    Survivors rendering window, or None if not found.

    Matches on WM_CLASS first (set from the exe name, stable across locales)
    over the whole tree, and only then on WM_NAME, so another window titled
    "Vampire Survivors" (e.g. a file manager open on the game folder) is not
    picked over the game.  getClientList() only covers WM-managed top-level
    windows; the actual game window is a child not registered there, so a
    full tree walk is required.
    """
    return _find_xid(win, lambda w: w.get_wm_class() == _VS_WM_CLASS) or _find_xid(
        win, lambda w: w.get_wm_name() == _VS_WM_NAME
    )


def _find_xid(win: XWindow, match) -> Optional[int]:
    """Depth-first search for the first window satisfying match."""
    try:
        if match(win):
            return win.id
    except Exception:
        pass
    try:
        children = win.query_tree().children
    except Exception:
        return None
    for child in children:
        result = _find_xid(child, match)
        if result is not None:
            return result
    return None


def get_abs_geometry(xwin: XWindow, root_id: int) -> tuple[int, int, int, int]:
    """Walk the X11 parent tree to convert window-relative coords to absolute screen coords.

    get_geometry() returns coords relative to the parent, not the root, so we
    accumulate offsets up the tree until we hit the root.  This mirrors the
    approach used internally by ewmhlib's _getWindowGeom helper.
    """
    geom = xwin.get_geometry()
    x, y, w, h = geom.x, geom.y, geom.width, geom.height
    win = xwin
    while True:
        parent = win.query_tree().parent
        if not isinstance(parent, XWindow):
            break
        pgeom = parent.get_geometry()
        x += pgeom.x
        y += pgeom.y
        if parent.id == root_id:
            break
        win = parent
    return x, y, w, h


def find_game_window(ewmh_root: EwmhRoot) -> GameWindow:
    """Locate the Vampire Survivors window and return a GameWindow.

    Uses a full X11 tree walk to find the rendering window by WM_CLASS /
    WM_NAME, matching the same logic as find_vs_xid().  The WM-managed
    top-level window visible in _NET_CLIENT_LIST is a Steam/Proton wrapper,
    not the actual rendering child — so _NET_CLIENT_LIST is skipped entirely.
    Raises RuntimeError if the window is not found.
    """
    xid = find_vs_xid(ewmh_root.root)
    if xid is None:
        raise RuntimeError("Vampire Survivors window not found — is the game running?")
    win = EwmhWindow(xid)
    pid = win.getPid() or -1
    x, y, w, h = get_abs_geometry(win.xWindow, ewmh_root.root.id)
    return GameWindow(xid=xid, pid=pid, x=x, y=y, width=w, height=h)


def send_input_to_game(
    display: Display, keys_to_release: list[int], keys_to_press: list[int]
) -> None:
    for k in keys_to_release:
        fake_input(display, KeyRelease, k)
    for k in keys_to_press:
        fake_input(display, KeyPress, k)
