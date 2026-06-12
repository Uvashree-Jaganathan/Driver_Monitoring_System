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
