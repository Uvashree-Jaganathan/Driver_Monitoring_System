import os
import sys

# --- 1. PATH FIX (MUST BE FIRST) ---
# This allows 'main.py' to see the 'utils' folder sitting outside 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import pygame
from landmarks import FaceLandmarks
from drowsiness import DrowsinessDetector 
from utils.eye_utils import get_eye_points, calculate_ear, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

# --- 2. SOUND SETUP (Using Sound object for better compatibility) ---
pygame.mixer.init()
current_dir = os.path.dirname(os.path.abspath(__file__))
alert_path = os.path.join(current_dir, "alert.wav.MP3") 

alert_sound = None
try:
    # Use Sound instead of music for better .wav/.mp3 compatibility
    alert_sound = pygame.mixer.Sound(alert_path)
    print(f"SUCCESS: Loaded sound from {alert_path}")
except Exception as e:
    print(f"ERROR: Could not load alert.wav: {e}")

def play_alert():
    if alert_sound:
        # Check if the sound is already playing on any channel
        if not pygame.mixer.get_busy():
            alert_sound.play(loops=-1) # Loops until stopped

def stop_alert():
    if alert_sound:
        alert_sound.stop()

# --- 3. INITIALIZE APPS ---
capture = cv2.VideoCapture(0)
face_app = FaceLandmarks()
# Threshold 0.22, Time 2.0 seconds
drowsy_app = DrowsinessDetector(threshold=0.22, drowsy_time=2.0)

while True:
    ret, frame = capture.read()
    if not ret: 
        break
    
    h, w, _ = frame.shape

    # 4. PROCESS LANDMARKS
    raw_points, results = face_app.process_frame(frame)

    if raw_points:
        # 5. CONVERT TO PIXELS
        points = np.array([(int(p[0]*w), int(p[1]*h)) for p in raw_points])
        
        # 6. CALCULATE EAR
        left_eye_pts = get_eye_points(points, LEFT_EYE_INDICES)
        right_eye_pts = get_eye_points(points, RIGHT_EYE_INDICES)
        
        current_ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0
        
        # 7. CHECK STATUS
        status, ear_val = drowsy_app.check_status(left_eye_pts, right_eye_pts)
        
        # DEBUG TERMINAL
        print(f"EAR: {current_ear:.2f} | Status: {status}")

        # --- 8. ALERT LOGIC (SOUND) ---
        if status == "DROWSY!":
            play_alert()
            status_color = (0, 0, 255) # Red
        else:
            stop_alert()
            status_color = (0, 255, 0) # Green

        # --- 9. VISUAL LOGIC ---
        # Draw Box
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), status_color, 2)

        # Draw Text
        cv2.putText(frame, f"STATUS: {status}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(frame, f"EAR: {ear_val:.2f}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 10. DISPLAY
    cv2.imshow("Driver Safety System", frame)
    
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27: 
        break

# CLEANUP
capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()