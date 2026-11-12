"""Phase 3: turn a flat 800x800 board image into an 8x8 grid of '.', 'w', 'b'.
Pure vision: no chess library is imported here."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2"))
from board_warp import SQ, square_slice

MARGIN = int(SQ * 0.2)      # skip 20% on every side -> analyze the center 60%
PIXEL_THR = 25              # gray difference that counts as "this pixel changed"
FRAC_THR = 0.15             # share of changed center pixels that means "occupied"
COLOR_SPLIT = 128           # piece pixels brighter than this -> white piece


def prep(warped):
    """Flat BGR board -> blurred gray (blur removes camera noise)."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def match_brightness(cur, ref):
    """Cancel a global lighting change: scale `cur` so it matches `ref` overall.
    The median of per-pixel ratios is robust, because pieces cover few pixels."""
    gain = np.median(cur.astype(np.float32) / (ref.astype(np.float32) + 1.0))
    gain = float(np.clip(gain, 0.5, 2.0))
    return np.clip(cur.astype(np.float32) / gain, 0, 255).astype(np.uint8)


def center(r, c):
    rs, cs = square_slice(r, c)
    return (slice(rs.start + MARGIN, rs.stop - MARGIN),
            slice(cs.start + MARGIN, cs.stop - MARGIN))


def read_grid(warped, ref_gray):
    """Compare the current flat board with the saved EMPTY board.
    Returns (grid, fracs): grid[r][c] in '.', 'w', 'b'; fracs[r][c] is the
    share of changed pixels, which is useful for tuning thresholds."""
    cur = match_brightness(prep(warped), ref_gray)
    diff = cv2.absdiff(cur, ref_gray)
    mask = (diff > PIXEL_THR).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop specks

    grid = [["."] * 8 for _ in range(8)]
    fracs = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            ctr = center(r, c)
            m = mask[ctr]
            fracs[r, c] = m.mean()
            if fracs[r, c] >= FRAC_THR:
                piece_pixels = cur[ctr][m > 0]
                grid[r][c] = "w" if piece_pixels.mean() > COLOR_SPLIT else "b"
    return grid, fracs


def grid_to_text(grid):
    """Print rank 8 at the top (assumes White at the bottom)."""
    lines = []
    for r in range(8):
        lines.append(f"{8 - r}  " + " ".join(grid[r]))
    lines.append("   a b c d e f g h")
    return "\n".join(lines)


def starting_grid():
    """What the start position should look like (White at the bottom)."""
    g = [["."] * 8 for _ in range(8)]
    for c in range(8):
        g[0][c], g[1][c] = "b", "b"
        g[6][c], g[7][c] = "w", "w"
    return g
