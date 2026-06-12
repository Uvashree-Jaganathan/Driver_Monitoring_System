import os
import sys

# --- 1. PATH FIX (MUST BE FIRST) ---
# This allows 'main.py' to see the project root when run as python src/main.py.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import pygame
from landmarks import FaceLandmarks
from drowsiness import DrowsinessDetector
from eye_classifier import EyeClassifier, crop_eye_region
from distraction import DistractionDetector
from src.temporal import TemporalState
from utils.eye_utils import get_eye_points, LEFT_EYE_INDICES, RIGHT_EYE_INDICES

# --- 2. SOUND SETUP ---
pygame.mixer.init()
current_dir = os.path.dirname(os.path.abspath(__file__))
alert_path = os.path.join(current_dir, "alert.wav.mp3")

alert_sound = None
try:
    alert_sound = pygame.mixer.Sound(alert_path)
    print(f"SUCCESS: Loaded sound from {alert_path}")
except Exception as e:
    print(f"ERROR: Could not load alert sound: {e}")


def play_alert():
    if alert_sound:
        if not pygame.mixer.get_busy():
            alert_sound.play(loops=-1)


def stop_alert():
    if alert_sound:
        alert_sound.stop()


# --- 3. INITIALIZE APPS ---
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

face_app = FaceLandmarks()
drowsy_app = DrowsinessDetector(threshold=0.22, drowsy_time=2.0)
eye_classifier = EyeClassifier()
cnn_drowsy_state = TemporalState(required_seconds=2.0)
distraction_app = DistractionDetector(distraction_time=2.0)

if eye_classifier.available:
    print("SUCCESS: Loaded CNN eye-state model.")
else:
    print("INFO: CNN model not found. Using EAR fallback for drowsiness.")

while True:
    ret, frame = capture.read()
    if not ret:
        break

    h, w, _ = frame.shape
    raw_points, results = face_app.process_frame(frame)

    if raw_points:
        points = np.array([(int(p[0] * w), int(p[1] * h)) for p in raw_points])

        left_eye_pts = get_eye_points(points, LEFT_EYE_INDICES)
        right_eye_pts = get_eye_points(points, RIGHT_EYE_INDICES)

        ear_status, ear_val = drowsy_app.check_status(left_eye_pts, right_eye_pts)
        status = ear_status
        cnn_label = "fallback"
        cnn_confidence = 0.0

        if eye_classifier.available:
            left_crop = crop_eye_region(frame, left_eye_pts)
            right_crop = crop_eye_region(frame, right_eye_pts)
            left_prediction = eye_classifier.predict_eye(left_crop)
            right_prediction = eye_classifier.predict_eye(right_crop)

            if left_prediction and right_prediction:
                closed_probability = (
                    left_prediction["closed_probability"]
                    + right_prediction["closed_probability"]
                ) / 2.0
                cnn_label = "closed" if closed_probability >= 0.5 else "open"
                cnn_confidence = (
                    closed_probability
                    if cnn_label == "closed"
                    else 1.0 - closed_probability
                )
                cnn_drowsy = cnn_drowsy_state.update(cnn_label == "closed")
                status = "DROWSY!" if cnn_drowsy else "ALERT"
            else:
                cnn_drowsy_state.update(False)
        else:
            cnn_drowsy_state.update(False)

        distraction_status, direction = distraction_app.check_status(points)

        if status == "DROWSY!":
            final_status = "DROWSY!"
        elif distraction_status == "DISTRACTED":
            final_status = "DISTRACTED"
        else:
            final_status = "ALERT"

        print(
            f"EAR: {ear_val:.2f} | CNN: {cnn_label} {cnn_confidence:.2f} | "
            f"Direction: {direction} | Status: {final_status}"
        )

        if final_status in ("DROWSY!", "DISTRACTED"):
            play_alert()
            status_color = (0, 0, 255)
        else:
            stop_alert()
            status_color = (0, 255, 0)

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), status_color, 2)

        cv2.putText(frame, f"STATUS: {final_status}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(frame, f"EAR: {ear_val:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"CNN: {cnn_label} {cnn_confidence:.2f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FACE: {direction}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        stop_alert()
        cv2.putText(frame, "STATUS: NO_FACE", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Driver Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
