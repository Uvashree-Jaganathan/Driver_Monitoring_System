# Driver Monitoring System

A real-time driver monitoring system that detects drowsiness and driver distraction using computer vision, facial landmarks, temporal analysis, and a PyTorch CNN eye-state classifier.

The system uses a webcam feed to track the driver's face, classify eye state, estimate distraction from face direction, and trigger low-latency visual/audio alerts when unsafe behavior is detected.

## Features

- Real-time webcam-based driver monitoring
- MediaPipe FaceMesh facial landmark tracking
- PyTorch CNN model for open/closed eye classification
- Eye Aspect Ratio fallback when no CNN model is available
- Temporal filtering to reduce false alerts
- Distraction detection from sustained face-direction changes
- Audio and visual alerts for drowsiness or distraction
- Training script for open/closed eye datasets
- Automated tests for core detection logic

## Tech Stack

- Python
- OpenCV
- MediaPipe
- PyTorch
- NumPy
- pygame
- torchvision
- pytest

## Project Structure

```text
Driver_Monitoring_System/
  src/
    main.py              # Real-time webcam application
    landmarks.py         # MediaPipe FaceMesh wrapper
    drowsiness.py        # EAR-based drowsiness fallback
    cnn_model.py         # PyTorch CNN architecture
    eye_classifier.py    # Eye crop preprocessing and CNN inference
    distraction.py       # Landmark-based distraction detection
    temporal.py          # Sustained-state timing helper
    alert.wav.mp3        # Alert sound
  utils/
    eye_utils.py         # Eye landmark and EAR utilities
  scripts/
    train_eye_cnn.py     # CNN training script
  tests/
    test_*.py            # Unit and smoke tests
  requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the default PyTorch install downloads large CUDA packages on Linux, install CPU-only PyTorch instead:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn pygame pytest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python src/main.py
```

Press `Esc` to exit.

If `models/eye_state_cnn.pt` exists, the app loads the trained CNN model. If the model is missing, the app continues using the EAR-based drowsiness fallback.

## CNN Training

The training script expects an open/closed eye dataset in this format:

```text
data/eye_state/
  train/
    open/
    closed/
  val/
    open/
    closed/
```

Train the model:

```bash
python scripts/train_eye_cnn.py --data-dir data/eye_state --output models/eye_state_cnn.pt --epochs 3 --batch-size 128
```

The trained model is saved to:

```text
models/eye_state_cnn.pt
```

## Training Result

The CNN was trained locally on an open/closed eye dataset mapped as:

- `awake` -> `open`
- `sleepy` -> `closed`

Training summary:

```text
epoch=1 train_acc=0.882 val_acc=0.942
epoch=2 train_acc=0.954 val_acc=0.969
epoch=3 train_acc=0.968 val_acc=0.974
```

The final validation accuracy was `97.4%`.

## Data And Model Files

Dataset and model artifacts are intentionally ignored by Git:

```text
data/
models/
```

This keeps the repository lightweight. To reproduce the trained CNN, download an open/closed eye dataset, arrange it using the folder layout above, and run the training command.

## Tests

Run the test suite:

```bash
python -m pytest -v
```

If your shell has ROS sourced and pytest tries to load ROS plugins, use:

```bash
env PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v
```

Current verification:

```text
14 passed
```

## How It Works

1. OpenCV captures frames from the webcam.
2. MediaPipe FaceMesh extracts facial landmarks.
3. Eye landmarks are used to crop both eyes from the frame.
4. The CNN classifies the eye state as open or closed.
5. EAR-based drowsiness detection is available as a fallback.
6. Face-direction geometry detects sustained looking away.
7. Temporal filtering confirms unsafe states before triggering alerts.
8. pygame plays an alert sound and OpenCV overlays status text on the frame.

## CV Summary

Designed a real-time driver monitoring system using OpenCV, MediaPipe, and PyTorch CNN-based eye-state classification to detect drowsiness and distraction. Implemented facial landmark tracking, temporal analysis, and a low-latency audio/visual alert system for webcam-based driving simulation.

## Limitations

- The CNN model file is not committed to GitHub.
- Runtime performance depends on webcam quality, lighting, and CPU/GPU capability.
- Distraction detection is landmark-geometry based, not a separate deep learning classifier.
- This project is for academic and demonstration use, not production vehicle safety deployment.
