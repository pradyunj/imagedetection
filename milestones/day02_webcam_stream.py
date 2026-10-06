# ==============================================================================
# Project: Raptor Defense System (Pradyun's AI)
# Day 2: Video Capture Logic (Live Webcam Feed & Loop Lifecycle)
# ==============================================================================

import cv2

print("[SYSTEM] Initializing camera interface...")

# Step 1: Open the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Step 2: Safety check - verify camera opened successfully
if not cap.isOpened():
    print("[ERROR] Could not connect to camera hardware (Index 0).")
    exit()

print("[SYSTEM] Camera connected successfully! Press 'q' to quit.")

# Step 3: Infinite loop to continuously read and display frames
while True:
    # Read frame from video capture device
    ret, frame = cap.read()
    
    # If frame grab fails, break the loop
    if not ret:
        print("[WARNING] Failed to grab frame from camera.")
        break
        
    # Overlay instructions
    cv2.putText(frame, "Day 2: Live Webcam Stream (Press 'q' to exit)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
    # Display the live frame
    cv2.imshow("Pradyun's AI - Live Webcam", frame)
    
    # Listen for 'q' key press to break out of loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[SYSTEM] 'q' pressed. Initiating graceful shutdown...")
        break

# Step 4: Release hardware resource and destroy windows
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Day 2 Complete! Camera released cleanly.")
