# CV-Matching Perception System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real PyTorch CNN inference path, MediaPipe-based distraction detection, temporal decision filtering, and documentation so the project matches the approved CV-facing design.

**Architecture:** Keep webcam and FaceMesh flow in `src/main.py`, but move reusable decision logic into focused modules. The CNN path loads `models/eye_state_cnn.pt` when present and falls back to the existing EAR detector when missing. Distraction detection uses landmark geometry and the same temporal confirmation pattern as drowsiness.

**Tech Stack:** Python, OpenCV, MediaPipe, NumPy, PyTorch, torchvision, pygame, pytest.

---

## File Structure

- Create `src/temporal.py`: shared sustained-condition timer used by drowsiness and distraction decisions.
- Modify `src/drowsiness.py`: use `TemporalState` while preserving the existing `DrowsinessDetector.check_status(left_eye, right_eye)` interface.
- Create `src/cnn_model.py`: compact binary CNN model for open/closed eye classification.
- Create `src/eye_classifier.py`: eye crop preprocessing, optional model loading, and CNN inference wrapper.
- Create `src/distraction.py`: face-direction classification and sustained distraction detection.
- Modify `src/main.py`: integrate CNN availability, distraction status, combined status, and clearer fallback messages.
- Create `scripts/train_eye_cnn.py`: training entry point for future public dataset use.
- Create `tests/`: focused unit tests for timing, CNN shape/preprocessing, and distraction logic.
- Create or modify `README.md`: project description, setup, run, model training, and honest CV wording.

---

### Task 1: Add Shared Temporal State And Update Drowsiness Tests

**Files:**
- Create: `src/temporal.py`
- Modify: `src/drowsiness.py`
- Create: `tests/test_temporal.py`
- Create: `tests/test_drowsiness.py`

- [ ] **Step 1: Write failing tests for temporal confirmation**

Create `tests/test_temporal.py`:

```python
from src.temporal import TemporalState


def test_temporal_state_waits_until_duration_is_met():
    state = TemporalState(required_seconds=2.0)

    assert state.update(True, now=10.0) is False
    assert state.update(True, now=11.5) is False
    assert state.update(True, now=12.0) is True


def test_temporal_state_resets_when_condition_clears:
    state = TemporalState(required_seconds=2.0)

    assert state.update(True, now=10.0) is False
    assert state.update(False, now=11.0) is False
    assert state.update(True, now=20.0) is False
    assert state.update(True, now=21.0) is False
```

Create `tests/test_drowsiness.py`:

```python
import numpy as np

from src.drowsiness import DrowsinessDetector


OPEN_LEFT = np.array([[0, 0], [1, 2], [2, 2], [4, 0], [2, -2], [1, -2]], dtype=float)
OPEN_RIGHT = OPEN_LEFT.copy()
CLOSED_LEFT = np.array([[0, 0], [1, 0.2], [2, 0.2], [4, 0], [2, -0.2], [1, -0.2]], dtype=float)
CLOSED_RIGHT = CLOSED_LEFT.copy()


def test_drowsiness_detector_uses_temporal_threshold():
    detector = DrowsinessDetector(threshold=0.22, drowsy_time=2.0)

    status, _ = detector.check_status(CLOSED_LEFT, CLOSED_RIGHT, now=100.0)
    assert status == "ALERT"

    status, _ = detector.check_status(CLOSED_LEFT, CLOSED_RIGHT, now=101.0)
    assert status == "ALERT"

    status, _ = detector.check_status(CLOSED_LEFT, CLOSED_RIGHT, now=102.0)
    assert status == "DROWSY!"


def test_drowsiness_detector_resets_when_eyes_open():
    detector = DrowsinessDetector(threshold=0.22, drowsy_time=2.0)

    detector.check_status(CLOSED_LEFT, CLOSED_RIGHT, now=100.0)
    detector.check_status(OPEN_LEFT, OPEN_RIGHT, now=101.0)
    status, _ = detector.check_status(CLOSED_LEFT, CLOSED_RIGHT, now=102.0)

    assert status == "ALERT"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_temporal.py tests/test_drowsiness.py -v
```

Expected: FAIL because `src.temporal` does not exist and `DrowsinessDetector.check_status()` does not accept `now`.

- [ ] **Step 3: Implement temporal helper**

Create `src/temporal.py`:

```python
import time


class TemporalState:
    def __init__(self, required_seconds):
        self.required_seconds = required_seconds
        self.start_time = None

    def update(self, condition_active, now=None):
        current_time = time.time() if now is None else now

        if not condition_active:
            self.start_time = None
            return False

        if self.start_time is None:
            self.start_time = current_time
            return False

        return current_time - self.start_time >= self.required_seconds
```

- [ ] **Step 4: Update drowsiness detector**

Replace `src/drowsiness.py` with:

```python
from utils.eye_utils import calculate_ear
from src.temporal import TemporalState


class DrowsinessDetector:
    def __init__(self, threshold=0.22, drowsy_time=2.0):
        self.threshold = threshold
        self.drowsy_time = drowsy_time
        self.temporal_state = TemporalState(required_seconds=drowsy_time)

    def check_status(self, left_eye, right_eye, now=None):
        ear_l = calculate_ear(left_eye)
        ear_r = calculate_ear(right_eye)
        avg_ear = (ear_l + ear_r) / 2.0

        is_closed = avg_ear < self.threshold
        is_drowsy = self.temporal_state.update(is_closed, now=now)
        status = "DROWSY!" if is_drowsy else "ALERT"

        return status, avg_ear
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/test_temporal.py tests/test_drowsiness.py -v
```

Expected: PASS.

Commit:

```bash
git add src/temporal.py src/drowsiness.py tests/test_temporal.py tests/test_drowsiness.py
git commit -m "test: add temporal drowsiness logic"
```

---

### Task 2: Add CNN Model And Eye Classifier

**Files:**
- Create: `src/cnn_model.py`
- Create: `src/eye_classifier.py`
- Create: `tests/test_eye_classifier.py`

- [ ] **Step 1: Write failing tests for CNN and classifier behavior**

Create `tests/test_eye_classifier.py`:

```python
import numpy as np
import torch

from src.cnn_model import EyeStateCNN
from src.eye_classifier import EyeClassifier, crop_eye_region, preprocess_eye_crop


def test_eye_state_cnn_outputs_two_logits():
    model = EyeStateCNN()
    batch = torch.zeros((2, 1, 32, 32), dtype=torch.float32)

    output = model(batch)

    assert output.shape == (2, 2)


def test_preprocess_eye_crop_returns_normalized_tensor():
    crop = np.full((20, 40, 3), 128, dtype=np.uint8)

    tensor = preprocess_eye_crop(crop)

    assert tensor.shape == (1, 1, 32, 32)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_crop_eye_region_expands_landmark_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    points = np.array([[40, 40], [50, 40], [50, 50], [40, 50]], dtype=int)

    crop = crop_eye_region(frame, points, padding=5)

    assert crop.shape == (20, 20, 3)


def test_eye_classifier_reports_unavailable_without_model_file(tmp_path):
    classifier = EyeClassifier(model_path=tmp_path / "missing.pt")

    assert classifier.available is False
    assert classifier.predict_eye(np.zeros((32, 32, 3), dtype=np.uint8)) is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_eye_classifier.py -v
```

Expected: FAIL because `src.cnn_model` and `src.eye_classifier` do not exist.

- [ ] **Step 3: Implement CNN model**

Create `src/cnn_model.py`:

```python
import torch.nn as nn


class EyeStateCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

- [ ] **Step 4: Implement eye classifier**

Create `src/eye_classifier.py`:

```python
from pathlib import Path

import cv2
import numpy as np
import torch

from src.cnn_model import EyeStateCNN


CLASS_NAMES = ("closed", "open")


def crop_eye_region(frame, eye_points, padding=8):
    x_min, y_min = np.min(eye_points, axis=0)
    x_max, y_max = np.max(eye_points, axis=0)

    height, width = frame.shape[:2]
    x_min = max(int(x_min) - padding, 0)
    y_min = max(int(y_min) - padding, 0)
    x_max = min(int(x_max) + padding, width)
    y_max = min(int(y_max) + padding, height)

    if x_max <= x_min or y_max <= y_min:
        return None

    return frame[y_min:y_max, x_min:x_max]


def preprocess_eye_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


class EyeClassifier:
    def __init__(self, model_path="models/eye_state_cnn.pt", device=None):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = EyeStateCNN().to(self.device)
        self.available = False

        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.available = True

    def predict_eye(self, crop):
        if not self.available or crop is None or crop.size == 0:
            return None

        tensor = preprocess_eye_crop(crop).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, index = torch.max(probabilities, dim=0)

        return {
            "label": CLASS_NAMES[int(index.item())],
            "confidence": float(confidence.item()),
            "closed_probability": float(probabilities[0].item()),
            "open_probability": float(probabilities[1].item()),
        }
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
pytest tests/test_eye_classifier.py -v
```

Expected: PASS.

Commit:

```bash
git add src/cnn_model.py src/eye_classifier.py tests/test_eye_classifier.py
git commit -m "feat: add eye state CNN inference module"
```

---

### Task 3: Add Distraction Detection

**Files:**
- Create: `src/distraction.py`
- Create: `tests/test_distraction.py`

- [ ] **Step 1: Write failing tests for face direction and temporal distraction**

Create `tests/test_distraction.py`:

```python
import numpy as np

from src.distraction import DistractionDetector, estimate_face_direction


def make_points(nose_x=50, nose_y=50):
    points = np.zeros((468, 2), dtype=float)
    points[33] = [30, 45]
    points[263] = [70, 45]
    points[152] = [50, 85]
    points[10] = [50, 15]
    points[1] = [nose_x, nose_y]
    return points


def test_estimate_face_direction_attentive():
    assert estimate_face_direction(make_points()) == "ATTENTIVE"


def test_estimate_face_direction_left_and_right():
    assert estimate_face_direction(make_points(nose_x=35)) == "LOOKING_LEFT"
    assert estimate_face_direction(make_points(nose_x=65)) == "LOOKING_RIGHT"


def test_estimate_face_direction_up_and_down():
    assert estimate_face_direction(make_points(nose_y=35)) == "LOOKING_UP"
    assert estimate_face_direction(make_points(nose_y=65)) == "LOOKING_DOWN"


def test_distraction_detector_requires_sustained_direction():
    detector = DistractionDetector(distraction_time=2.0)
    points = make_points(nose_x=65)

    status, direction = detector.check_status(points, now=10.0)
    assert status == "ALERT"
    assert direction == "LOOKING_RIGHT"

    status, direction = detector.check_status(points, now=12.0)
    assert status == "DISTRACTED"
    assert direction == "LOOKING_RIGHT"


def test_distraction_detector_resets_when_attentive():
    detector = DistractionDetector(distraction_time=2.0)

    detector.check_status(make_points(nose_x=65), now=10.0)
    detector.check_status(make_points(), now=11.0)
    status, _ = detector.check_status(make_points(nose_x=65), now=12.0)

    assert status == "ALERT"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_distraction.py -v
```

Expected: FAIL because `src.distraction` does not exist.

- [ ] **Step 3: Implement distraction detection**

Create `src/distraction.py`:

```python
from src.temporal import TemporalState


NO_FACE = "NO_FACE"
ATTENTIVE = "ATTENTIVE"
LOOKING_LEFT = "LOOKING_LEFT"
LOOKING_RIGHT = "LOOKING_RIGHT"
LOOKING_UP = "LOOKING_UP"
LOOKING_DOWN = "LOOKING_DOWN"


def estimate_face_direction(points, horizontal_threshold=0.18, vertical_threshold=0.22):
    if points is None or len(points) <= 263:
        return NO_FACE

    left_eye_outer = points[33]
    right_eye_outer = points[263]
    face_top = points[10]
    face_bottom = points[152]
    nose_tip = points[1]

    eye_width = max(abs(right_eye_outer[0] - left_eye_outer[0]), 1.0)
    face_height = max(abs(face_bottom[1] - face_top[1]), 1.0)
    eye_center_x = (left_eye_outer[0] + right_eye_outer[0]) / 2.0
    face_center_y = (face_top[1] + face_bottom[1]) / 2.0

    horizontal_offset = (nose_tip[0] - eye_center_x) / eye_width
    vertical_offset = (nose_tip[1] - face_center_y) / face_height

    if horizontal_offset <= -horizontal_threshold:
        return LOOKING_LEFT
    if horizontal_offset >= horizontal_threshold:
        return LOOKING_RIGHT
    if vertical_offset <= -vertical_threshold:
        return LOOKING_UP
    if vertical_offset >= vertical_threshold:
        return LOOKING_DOWN
    return ATTENTIVE


class DistractionDetector:
    def __init__(self, distraction_time=2.0):
        self.temporal_state = TemporalState(required_seconds=distraction_time)

    def check_status(self, points, now=None):
        direction = estimate_face_direction(points)
        is_distracted = direction not in (ATTENTIVE, NO_FACE)
        sustained = self.temporal_state.update(is_distracted, now=now)
        status = "DISTRACTED" if sustained else "ALERT"
        return status, direction
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
pytest tests/test_distraction.py -v
```

Expected: PASS.

Commit:

```bash
git add src/distraction.py tests/test_distraction.py
git commit -m "feat: add temporal distraction detection"
```

---

### Task 4: Add Future CNN Training Script

**Files:**
- Create: `scripts/train_eye_cnn.py`
- Create: `tests/test_training_script.py`

- [ ] **Step 1: Write failing smoke test**

Create `tests/test_training_script.py`:

```python
import importlib.util
from pathlib import Path


def test_training_script_exists_and_exposes_main():
    script_path = Path("scripts/train_eye_cnn.py")
    spec = importlib.util.spec_from_file_location("train_eye_cnn", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "main")
    assert callable(module.main)
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_training_script.py -v
```

Expected: FAIL because `scripts/train_eye_cnn.py` does not exist.

- [ ] **Step 3: Implement training script**

Create `scripts/train_eye_cnn.py`:

```python
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.cnn_model import EyeStateCNN


def build_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(Path(data_dir) / "train", transform=transform)
    val_dataset = datasets.ImageFolder(Path(data_dir) / "val", transform=transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        train_dataset.class_to_idx,
    )


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train eye-state CNN on open/closed eye images.")
    parser.add_argument("--data-dir", default="data/eye_state")
    parser.add_argument("--output", default="models/eye_state_cnn.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = build_dataloaders(args.data_dir, args.batch_size)

    if set(class_to_idx.keys()) != {"closed", "open"}:
        raise ValueError("Dataset folders must be named exactly 'closed' and 'open'.")

    model = EyeStateCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"saved_model={output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test and commit**

Run:

```bash
pytest tests/test_training_script.py -v
```

Expected: PASS.

Commit:

```bash
git add scripts/train_eye_cnn.py tests/test_training_script.py
git commit -m "feat: add eye CNN training script"
```

---

### Task 5: Integrate CNN And Distraction Detection Into Main Loop

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Make imports package-safe**

Update imports in `src/main.py` so the file can still be run as `python src/main.py`:

```python
from eye_classifier import EyeClassifier, crop_eye_region
from distraction import DistractionDetector
```

Keep the existing path fix at the top before these imports.

- [ ] **Step 2: Initialize new detectors**

After the current detector initialization, add:

```python
eye_classifier = EyeClassifier()
distraction_app = DistractionDetector(distraction_time=2.0)

if eye_classifier.available:
    print("SUCCESS: Loaded CNN eye-state model.")
else:
    print("INFO: CNN model not found. Using EAR fallback for drowsiness.")
```

- [ ] **Step 3: Add combined detection logic inside `if raw_points:` block**

Replace the current status calculation and alert decision block with:

```python
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
            left_prediction["closed_probability"] + right_prediction["closed_probability"]
        ) / 2.0
        cnn_label = "closed" if closed_probability >= 0.5 else "open"
        cnn_confidence = closed_probability if cnn_label == "closed" else 1.0 - closed_probability
        if cnn_label == "closed":
            status = "DROWSY!"
        else:
            status = "ALERT"

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
```

- [ ] **Step 4: Update overlay text**

Replace the existing text overlay with:

```python
cv2.putText(frame, f"STATUS: {final_status}", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
cv2.putText(frame, f"EAR: {ear_val:.2f}", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(frame, f"CNN: {cnn_label} {cnn_confidence:.2f}", (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(frame, f"FACE: {direction}", (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

- [ ] **Step 5: Add no-face alert reset**

Add an `else` for `if raw_points:` before display:

```python
else:
    stop_alert()
    cv2.putText(frame, "STATUS: NO_FACE", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
```

- [ ] **Step 6: Run import checks**

Run:

```bash
python -m py_compile src/main.py src/cnn_model.py src/eye_classifier.py src/distraction.py src/drowsiness.py src/temporal.py
```

Expected: exits with code 0.

- [ ] **Step 7: Commit**

Commit:

```bash
git add src/main.py
git commit -m "feat: integrate CNN and distraction monitoring"
```

---

### Task 6: Update README And Final Verification

**Files:**
- Create or modify: `README.md`

- [ ] **Step 1: Write README**

Create or replace `README.md`:

```markdown
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

## CV Wording

This project supports the wording:

> Designed a real-time driver monitoring system using OpenCV, MediaPipe, and a PyTorch CNN inference pipeline to detect drowsiness and distraction, with facial landmark tracking, temporal analysis, and low-latency alert feedback.

For a stronger CNN training claim, train `models/eye_state_cnn.pt` using a real open/closed-eye dataset.
```

- [ ] **Step 2: Run full automated verification**

Run:

```bash
pytest -v
python -m py_compile src/main.py src/cnn_model.py src/eye_classifier.py src/distraction.py src/drowsiness.py src/temporal.py scripts/train_eye_cnn.py
```

Expected: all tests pass and py_compile exits with code 0.

- [ ] **Step 3: Manual verification**

Run:

```bash
python src/main.py
```

Expected:

- Console prints either `SUCCESS: Loaded CNN eye-state model.` or `INFO: CNN model not found. Using EAR fallback for drowsiness.`
- Webcam window opens.
- Looking at camera shows `STATUS: ALERT`.
- Closing eyes for about 2 seconds shows `STATUS: DROWSY!`.
- Looking left or right for about 2 seconds shows `STATUS: DISTRACTED`.
- Pressing `Esc` exits cleanly.

- [ ] **Step 4: Commit**

Commit:

```bash
git add README.md
git commit -m "docs: document CV-matching monitoring system"
```

---

## Final Review

- [ ] Run `git status --short` and confirm only intended files are modified.
- [ ] Run `git log --oneline -5` and confirm task commits are present.
- [ ] Report final test results, manual verification status, and whether CNN is running with real weights or EAR fallback.
