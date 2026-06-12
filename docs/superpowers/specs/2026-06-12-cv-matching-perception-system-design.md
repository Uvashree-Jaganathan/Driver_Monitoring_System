# CV-Matching Driver Monitoring System Design

## Goal

Upgrade the current EAR-based driver drowsiness project so it honestly supports the CV description: a real-time perception and decision system using OpenCV, MediaPipe, and a PyTorch CNN inference pipeline to detect drowsiness and distraction with temporal filtering and low-latency alerts.

The project will not pretend to include a fully trained production model without data. Instead, it will include the CNN architecture, loading/inference path, training script, and fallback behavior. A trained model can be added later from a public open/closed-eye dataset.

## Current State

The project currently has:

- Webcam capture and real-time display in `src/main.py`.
- MediaPipe FaceMesh landmark extraction in `src/landmarks.py`.
- Eye Aspect Ratio drowsiness logic with a 2-second threshold in `src/drowsiness.py`.
- Audio and visual alert output in `src/main.py`.

The project does not currently have:

- Any PyTorch model implementation or inference.
- Any trained CNN weights.
- Any distraction detection.
- Any training or evaluation script.

## Proposed Architecture

Add a small set of focused modules:

- `src/cnn_model.py`: defines a compact PyTorch CNN for binary eye-state classification: open vs closed.
- `src/eye_classifier.py`: crops eye regions from the frame using MediaPipe landmarks, preprocesses them, loads a saved model if available, and returns CNN predictions.
- `src/distraction.py`: detects driver distraction using landmark-derived face orientation signals such as nose position relative to face bounds and eye/face geometry. It reports looking left, right, up, down, or attentive.
- `scripts/train_eye_cnn.py`: trains the CNN from a folder dataset when data is available. Expected folder layout is `data/eye_state/train/open`, `data/eye_state/train/closed`, `data/eye_state/val/open`, and `data/eye_state/val/closed`.
- `README.md`: documents setup, runtime behavior, model training, and what claims the project supports.

Keep the existing modules where possible. The main loop should coordinate detectors rather than contain all detection logic.

## Runtime Data Flow

1. Read webcam frame with OpenCV.
2. Extract MediaPipe FaceMesh landmarks.
3. Convert normalized landmarks to pixel coordinates.
4. Run the existing EAR detector as a reliable fallback.
5. If `models/eye_state_cnn.pt` exists, crop both eyes and run CNN inference.
6. Combine eye predictions over time to decide drowsiness.
7. Run distraction detection from face orientation landmarks.
8. Trigger alert if the driver is drowsy or distracted for the configured temporal threshold.
9. Draw status, EAR/CNN confidence, distraction state, and face box on the frame.

## Decision Logic

Drowsiness:

- If CNN model is available, use CNN closed-eye predictions as the primary signal.
- If CNN model is missing or inference fails, use the current EAR-based detector.
- Require sustained closed-eye state for the configured threshold before declaring `DROWSY`.

Distraction:

- Estimate face direction from stable facial landmark geometry.
- Classify large sustained deviations as `DISTRACTED_LEFT`, `DISTRACTED_RIGHT`, `DISTRACTED_UP`, or `DISTRACTED_DOWN`.
- Require sustained distraction for the configured threshold before declaring `DISTRACTED`.

Alerts:

- Play the existing alert sound when final status is `DROWSY` or `DISTRACTED`.
- Stop sound when final status returns to `ALERT`.
- Keep visual status overlays for real-time feedback.

## Error Handling

- If the CNN model file is missing, print a clear message and continue with EAR fallback.
- If camera access fails, exit cleanly with a readable error.
- If no face is detected, show `NO_FACE` and stop alerts.
- If audio loading fails, continue with visual alerts only.
- If an eye crop is invalid, skip CNN for that frame and use EAR fallback.

## Testing And Verification

Add focused checks:

- Unit tests for EAR threshold timing behavior.
- Unit tests for distraction temporal thresholding with synthetic landmark-like points.
- A lightweight import/smoke test for the CNN model and classifier preprocessing.
- Manual webcam verification:
  - Eyes open shows `ALERT`.
  - Eyes closed for the threshold shows `DROWSY`.
  - Looking away for the threshold shows `DISTRACTED`.
  - Missing CNN model falls back to EAR without crashing.

## CV Wording Supported After Implementation

After this design is implemented, the CV can honestly say:

> Designed a real-time driver monitoring system using OpenCV, MediaPipe, and a PyTorch CNN inference pipeline to detect drowsiness and distraction, with facial landmark tracking, temporal analysis, and low-latency alert feedback.

If a public dataset is later used to train `models/eye_state_cnn.pt`, the CV can more strongly mention CNN-based drowsiness detection.
