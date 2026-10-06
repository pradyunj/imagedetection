# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 1: Environment Setup & OpenCV Blank Canvas
# ==============================================================================

# Step 1: Import the newly installed computer vision library
import cv2
import numpy as np

# Step 2: Print welcoming status message
print("[SYSTEM] Starting Pradyun's AI...")
print("[SYSTEM] Initializing blank canvas window...")

# Step 3: Create a clean blank digital image matrix (480x640, 3 color channels)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Step 4: Add welcome text onto the canvas for visual feedback
cv2.putText(
    canvas,
    "Pradyun's AI: Day 1 - Blank Canvas",
    (50, 220),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
)
cv2.putText(
    canvas,
    "Press any key to close the window",
    (50, 260),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    1,
    cv2.LINE_AA,
)

# Step 5: Open GUI display window named "Pradyun's AI"
cv2.imshow("Pradyun's AI", canvas)
print("[SYSTEM] Window opened. Waiting for keypress to exit...")

# Step 6: Hold window open until a key is pressed
cv2.waitKey(0)

# Step 7: Cleanly close all windows
cv2.destroyAllWindows()
print("[SYSTEM] Day 1 Complete! Window closed cleanly.")
