"""Phase 3 self-test (no camera): simulate a tilted camera photo of a board with
the starting position, with noise and lighting changes, then read it back.
Run from the repo root:  python3 phase3/phase3_selftest.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "phase2"))
from board_reader import grid_to_text, prep, read_grid, starting_grid
from board_warp import SIZE, SQ, get_matrix, warp

LIGHT, DARK = 190, 100                  # square brightness
WHITE_P, BLACK_P = 245, 30              # piece brightness
FAKE_CORNERS = [(300, 120), (900, 160), (1000, 640), (200, 600)]
TO_CAMERA = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]]), np.float32(FAKE_CORNERS))
MATRIX = get_matrix(FAKE_CORNERS)


def photo(grid, gain=1.0, noise=3, seed=0):
    """Draw a board with pieces, put it through a fake camera."""
    flat = np.zeros((SIZE, SIZE, 3), np.uint8)
    for r in range(8):
        for c in range(8):
            v = LIGHT if (r + c) % 2 == 0 else DARK
            flat[r*SQ:(r+1)*SQ, c*SQ:(c+1)*SQ] = v
            if grid[r][c] != ".":
                v = WHITE_P if grid[r][c] == "w" else BLACK_P
                cv2.circle(flat, (c*SQ + SQ//2, r*SQ + SQ//2), 30, (v, v, v), -1)
    cam = cv2.warpPerspective(flat, TO_CAMERA, (1280, 720), borderValue=(50, 50, 50))
    cam = cam.astype(np.float32) * gain
    cam += np.random.default_rng(seed).normal(0, noise, cam.shape)
    return np.clip(cam, 0, 255).astype(np.uint8)


empty = [["."] * 8 for _ in range(8)]
ref = prep(warp(photo(empty, seed=99), MATRIX))        # the saved empty-board reference

# 1. Starting position, 5 reads in a row (different noise each time)
for i in range(5):
    g, _ = read_grid(warp(photo(starting_grid(), seed=i), MATRIX), ref)
    assert g == starting_grid(), f"read {i} differs"
print("start position read correctly 5 times in a row")
print(grid_to_text(g))

# 2. A global lighting change (dimmer and brighter room)
for gain in (0.8, 1.15):
    g, _ = read_grid(warp(photo(starting_grid(), gain=gain, seed=7), MATRIX), ref)
    assert g == starting_grid(), f"lighting gain {gain} broke it"
print("lighting x0.8 and x1.15 OK (brightness matching works)")

# 3. After 1.e4: e2 empties, e4 gets a white piece
moved = [row[:] for row in starting_grid()]
moved[6][4], moved[4][4] = ".", "w"
g, _ = read_grid(warp(photo(moved, seed=3), MATRIX), ref)
assert g == moved
print("1.e4 detected: e2 empty, e4 white")
