"""Phase 1 study exercises: no camera needed. Run it and read the output."""
import cv2
import numpy as np

# --- A. An image is just a NumPy array ---
img = np.zeros((480, 640, 3), dtype=np.uint8)   # height, width, channels
img[:] = (255, 0, 0)                            # BGR! (255,0,0) is BLUE, not red
print("shape (h,w,c):", img.shape, "dtype:", img.dtype)

# slicing = cropping. Order is [y1:y2, x1:x2]  (rows first!)
crop = img[100:200, 300:500]
print("crop shape:", crop.shape)                # (100, 200, 3)
img[100:200, 300:500] = (0, 255, 0)             # paint the region green
print("pixel at x=350,y=150 (BGR):", img[150, 350])   # index is [y, x]

# --- B. Color spaces ---
bgr = np.full((1, 1, 3), (40, 180, 220), np.uint8)       # an orange-ish pixel
for name, code in [("GRAY", cv2.COLOR_BGR2GRAY),
                   ("HSV", cv2.COLOR_BGR2HSV),
                   ("LAB", cv2.COLOR_BGR2LAB)]:
    print(name, cv2.cvtColor(bgr, code)[0, 0])
# GRAY: one brightness number. Best first choice for "is this square occupied?"
# HSV : V = brightness, S = colorfulness. Helps vs shadows/glare (Phase 6).
# LAB : L = lightness, perceptually even. Handy for light-vs-dark pieces.

# --- C. Drawing (every call draws IN PLACE on the array) ---
canvas = np.full((400, 400, 3), 255, np.uint8)
cv2.line(canvas, (0, 0), (399, 399), (0, 0, 0), 2)
cv2.rectangle(canvas, (50, 50), (150, 150), (0, 0, 255), 2)       # (x,y) top-left, bottom-right
cv2.circle(canvas, (300, 100), 30, (0, 160, 0), -1)               # -1 = filled
cv2.arrowedLine(canvas, (100, 300), (300, 300), (255, 0, 0), 3)   # later: engine move arrow
cv2.putText(canvas, "e7e5", (130, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imwrite("drawing_demo.png", canvas)
print("wrote drawing_demo.png")

# --- D. Exercise for you (write the answers yourself, then check) ---
# 1. Crop the bottom-right 100x100 corner of `img` and print its shape.
# 2. Convert `canvas` to gray and print its shape. Why did the last number disappear?
# 3. Draw an 8x8 grid on a 400x400 white canvas (hint: loop i in range(9), step 50).
