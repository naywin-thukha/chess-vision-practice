"""Phase 2 mini-project: click four corners, see a flat 800x800 board with a grid.

Keys:  c = redo corners   g = grid on/off   s = save calibration   q = quit
Run from the repo root:   python3 phase2/phase2_calibrate.py [camera index or URL]
Click the 4 OUTER corners of the 8x8 playing area, in the order
top-left, top-right, bottom-right, bottom-left (as the camera sees them).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from phase1_capture import LatestFrame          # reuse the lag-free reader
from board_warp import (draw_grid, get_matrix, load_calibration, order_ok,
                        save_calibration, warp)

corners = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    stream = LatestFrame(int(arg) if arg.isdigit() else arg)
    if not stream.running:
        sys.exit("Could not open the video source")

    matrix = None
    loaded = load_calibration()
    if loaded:                                   # reuse last run's corners
        corners.extend(tuple(p) for p in loaded[0])
        matrix = loaded[1]
        print("Loaded saved calibration")

    show_grid = True
    cv2.namedWindow("camera")
    cv2.setMouseCallback("camera", on_mouse)

    while True:
        frame = stream.read()
        if frame is None:
            cv2.waitKey(10)
            continue

        view = frame.copy()
        for p in corners:
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
        if len(corners) == 4:
            if matrix is None:
                if order_ok(corners):
                    matrix = get_matrix(corners)
                else:
                    print("Corner order looks wrong. Press c and click TL, TR, BR, BL.")
                    corners.clear()
            cv2.polylines(view, [np.array(corners, np.int32)], True, (0, 255, 0), 2)
        cv2.imshow("camera", view)

        if matrix is not None:
            flat = warp(frame, matrix)
            if show_grid:
                draw_grid(flat)
            cv2.imshow("flat board", flat)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            corners.clear()
            matrix = None
            cv2.destroyWindow("flat board")
        elif key == ord("g"):
            show_grid = not show_grid
        elif key == ord("s") and matrix is not None:
            save_calibration(corners)
            print("Saved calibration.json")

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
