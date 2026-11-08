"""Phase 2 self-test: no camera needed. Fakes a tilted board photo, warps it back,
and checks that every square lands where it should.
Run from the repo root:  python3 phase2/phase2_selftest.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from board_warp import (SIZE, SQ, draw_grid, get_matrix, order_ok,
                        square_from_pixel, square_name, warp)

# 1. Draw a flat checkerboard, a1 dark (bottom-left), White at the bottom.
flat = np.zeros((SIZE, SIZE, 3), np.uint8)
for r in range(8):
    for c in range(8):
        light = (r + c) % 2 == 0          # a8 (top-left) is a light square
        flat[r*SQ:(r+1)*SQ, c*SQ:(c+1)*SQ] = (200, 220, 240) if light else (60, 90, 120)

# 2. Pretend a camera looked at it from an angle: squash it into a 1280x720 frame.
fake_corners = [(300, 120), (900, 160), (1000, 640), (200, 600)]   # TL TR BR BL
back = cv2.getPerspectiveTransform(np.float32([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]]),
                                   np.float32(fake_corners))
camera = cv2.warpPerspective(flat, back, (1280, 720))
cv2.imwrite("selftest_camera.png", camera)

# 3. Warp it back using only the four "clicked" corners.
assert order_ok(fake_corners)
M = get_matrix(fake_corners)
recovered = warp(camera, M)
cv2.imwrite("selftest_flat_grid.png", draw_grid(recovered.copy()))

# 4. Check: mean color of each recovered square matches the original.
worst = 0
for r in range(8):
    for c in range(8):
        a = recovered[r*SQ+30:(r+1)*SQ-30, c*SQ+30:(c+1)*SQ-30].mean(axis=(0, 1))
        b = flat[r*SQ+30:(r+1)*SQ-30, c*SQ+30:(c+1)*SQ-30].mean(axis=(0, 1))
        worst = max(worst, float(np.abs(a - b).max()))
print("worst color error over 64 squares:", round(worst, 1), "(should be small)")
assert worst < 25

# 5. Square naming checks (White at the bottom).
assert square_name(7, 0) == "a1" and square_name(0, 7) == "h8"
assert square_name(4, 4) == "e4" and square_name(6, 4) == "e2"
assert square_from_pixel(450, 450) == "e4"
print("square naming OK. Wrote selftest_camera.png and selftest_flat_grid.png")
