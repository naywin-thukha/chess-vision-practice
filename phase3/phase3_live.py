"""Phase 3 mini-project: print an 8x8 grid of empty / white / black squares.

Needs calibration.json from Phase 2 (run phase2_calibrate.py, press s).
Keys:  e = capture EMPTY board (do this first, no pieces!)
       r = read the board and print the grid
       t = print the changed-pixel share per square (for tuning thresholds)
       q = quit
Run from the repo root:  python3 phase3/phase3_live.py [camera index or URL]
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "phase1"))
sys.path.insert(0, os.path.join(HERE, "..", "phase2"))
from board_reader import grid_to_text, prep, read_grid, starting_grid
from board_warp import SQ, draw_grid, load_calibration, warp
from phase1_capture import LatestFrame

REF_FILE = "reference.png"


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    loaded = load_calibration()
    if not loaded:
        sys.exit("No calibration.json here. Run phase2/phase2_calibrate.py first (press s).")
    matrix = loaded[1]

    stream = LatestFrame(int(arg) if arg.isdigit() else arg)
    if not stream.running:
        sys.exit("Could not open the video source")

    ref = None
    if os.path.exists(REF_FILE):
        ref = prep(cv2.imread(REF_FILE))
        print("Loaded", REF_FILE)
    last_grid = None

    while True:
        frame = stream.read()
        if frame is None:
            cv2.waitKey(10)
            continue
        flat = warp(frame, matrix)
        view = flat.copy()
        draw_grid(view, labels=False)
        if last_grid:                                  # overlay W / B letters
            for r in range(8):
                for c in range(8):
                    ch = last_grid[r][c]
                    if ch != ".":
                        cv2.putText(view, ch.upper(), (c * SQ + 35, r * SQ + 62),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
        cv2.imshow("flat board", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            cv2.imwrite(REF_FILE, flat)
            ref = prep(flat)
            print("Saved empty-board reference ->", REF_FILE)
        elif key in (ord("r"), ord("t")):
            if ref is None:
                print("Press e first, with an empty board.")
                continue
            last_grid, fracs = read_grid(flat, ref)
            if key == ord("r"):
                print(grid_to_text(last_grid))
                print("matches starting position:", last_grid == starting_grid(), "\n")
            else:
                np.set_printoptions(precision=2, suppress=True)
                print(fracs, "\n")

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
