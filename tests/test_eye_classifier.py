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
