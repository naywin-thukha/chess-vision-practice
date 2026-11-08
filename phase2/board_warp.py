"""Phase 2 building blocks: perspective warp, saved calibration, square naming.
No camera code here, so it can be tested with plain images."""
import json
import os

import cv2
import numpy as np

SIZE = 800            # warped board is SIZE x SIZE pixels
SQ = SIZE // 8        # 100 px per square
CALIB_FILE = "calibration.json"

# Which side of the *warped* image is White's? True = White at the bottom,
# so the top-left of the warped image is a8 and the bottom-left is a1.
WHITE_AT_BOTTOM = True

# Destination corners, same order as the clicks: TL, TR, BR, BL
DST = np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]])


def order_ok(pts):
    """Sanity check for click order TL, TR, BR, BL (image y grows downward)."""
    tl, tr, br, bl = pts
    return tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]


def get_matrix(corners):
    """4 clicked points (TL, TR, BR, BL) -> 3x3 perspective matrix."""
    src = np.float32(corners)
    return cv2.getPerspectiveTransform(src, DST)


def warp(frame, matrix):
    """Camera frame -> flat top-down SIZE x SIZE board image."""
    return cv2.warpPerspective(frame, matrix, (SIZE, SIZE))


def save_calibration(corners, path=CALIB_FILE):
    with open(path, "w") as f:
        json.dump({"corners": [list(map(float, p)) for p in corners]}, f)


def load_calibration(path=CALIB_FILE):
    """Return (corners, matrix) or None if no file yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        corners = json.load(f)["corners"]
    return corners, get_matrix(corners)


def square_name(row, col):
    """Warped-image cell (row 0..7 from top, col 0..7 from left) -> 'e4'."""
    if WHITE_AT_BOTTOM:
        file_, rank = col, 7 - row
    else:                       # White at the top: board is rotated 180 degrees
        file_, rank = 7 - col, row
    return "abcdefgh"[file_] + str(rank + 1)


def square_from_pixel(x, y):
    """Pixel on the warped image -> square name."""
    return square_name(int(y) // SQ, int(x) // SQ)


def square_slice(row, col):
    """Pixel region of one square on the warped image."""
    return slice(row * SQ, (row + 1) * SQ), slice(col * SQ, (col + 1) * SQ)


def draw_grid(img, labels=True):
    """Draw the 8x8 grid (and square names) on a warped image, in place."""
    for i in range(9):
        cv2.line(img, (i * SQ, 0), (i * SQ, SIZE), (0, 255, 0), 1)
        cv2.line(img, (0, i * SQ), (SIZE, i * SQ), (0, 255, 0), 1)
    if labels:
        for r in range(8):
            for c in range(8):
                cv2.putText(img, square_name(r, c), (c * SQ + 4, r * SQ + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img
