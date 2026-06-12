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
