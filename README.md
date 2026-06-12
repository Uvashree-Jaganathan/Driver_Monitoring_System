# Driver Monitoring System

Real-time driver monitoring system using OpenCV, MediaPipe FaceMesh, NumPy, pygame, and a PyTorch CNN inference pipeline.

## Features

- Webcam-based real-time monitoring.
- MediaPipe facial landmark tracking.
- Eye Aspect Ratio fallback drowsiness detection.
- Optional PyTorch CNN inference for open/closed eye classification.
- Landmark-based distraction detection for sustained looking left, right, up, or down.
- Temporal filtering to reduce false alerts.
- Audio and visual alerts.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CPU-only PyTorch installs on Linux, use the official CPU wheel index if the default PyPI install starts downloading CUDA packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python src/main.py
```

Press `Esc` to exit.

If `models/eye_state_cnn.pt` is missing, the app prints a message and uses EAR fallback for drowsiness.

## CNN Training

Prepare data with this layout:

```text
data/eye_state/
  train/
    closed/
    open/
  val/
    closed/
    open/
```

Train:

```bash
python scripts/train_eye_cnn.py --data-dir data/eye_state --output models/eye_state_cnn.pt
```

## Tests

If your shell has ROS sourced and pytest tries to load ROS plugins, clear `PYTHONPATH` and disable external pytest plugin autoload for this project:

```bash
env PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v
```

## CV Wording

This project supports the wording:

> Designed a real-time driver monitoring system using OpenCV, MediaPipe, and a PyTorch CNN inference pipeline to detect drowsiness and distraction, with facial landmark tracking, temporal analysis, and low-latency alert feedback.

For a stronger CNN training claim, train `models/eye_state_cnn.pt` using a real open/closed-eye dataset.
