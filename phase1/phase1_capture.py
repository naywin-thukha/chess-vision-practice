"""Phase 1 mini-project: live phone video + click four board corners.

Keys:  c = (re)click corners   s = save frame   q = quit
Usage: python phase1_capture.py            (camera index 0)
       python phase1_capture.py 1          (another camera index, e.g. DroidCam/Iriun)
       python phase1_capture.py http://192.168.1.5:8080/video   (IP-camera URL)
"""
import sys
import threading
import time

import cv2
import numpy as np


# ---------- 1. Latest-frame reader (fixes stale-frame lag) ----------
class LatestFrame:
    """Reads the stream in a background thread and keeps ONLY the newest frame.
    Without this, cap.read() hands you old buffered frames and the video lags."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # not honored by every backend
        self.lock = threading.Lock()
        self.frame = None
        self.running = self.cap.isOpened()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


# ---------- 2. Mouse callback: collect 4 corners ----------
corners = []  # order matters: TL, TR, BR, BL


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))


# ---------- 3. Main loop ----------
def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(arg) if arg.isdigit() else arg
    stream = LatestFrame(source)
    if not stream.running:
        sys.exit("Could not open the video source: " + str(source))

    cv2.namedWindow("chess")
    cv2.setMouseCallback("chess", on_mouse)
    labels = ["TL", "TR", "BR", "BL"]
    n_saved = 0

    while True:
        frame = stream.read()
        if frame is None:
            cv2.waitKey(10)
            continue

        view = frame.copy()
        # draw clicked corners and the outline
        for i, p in enumerate(corners):
            cv2.circle(view, p, 6, (0, 0, 255), -1)
            cv2.putText(view, labels[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(corners) == 4:
            cv2.polylines(view, [np.array(corners, np.int32)], True, (0, 255, 0), 2)
            hint = "Corners set. c=redo  s=save  q=quit"
        else:
            hint = f"Click corner {len(corners) + 1}/4: {labels[len(corners)]}"
        cv2.putText(view, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("chess", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            corners.clear()
        elif key == ord("s"):
            name = f"frame_{n_saved:03d}.png"
            cv2.imwrite(name, frame)  # save the clean frame, without drawings
            print("saved", name, frame.shape)
            n_saved += 1

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
