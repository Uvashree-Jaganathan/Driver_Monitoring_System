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
