from src.temporal import TemporalState


NO_FACE = "NO_FACE"
ATTENTIVE = "ATTENTIVE"
LOOKING_LEFT = "LOOKING_LEFT"
LOOKING_RIGHT = "LOOKING_RIGHT"
LOOKING_UP = "LOOKING_UP"
LOOKING_DOWN = "LOOKING_DOWN"


def estimate_face_direction(points, horizontal_threshold=0.18, vertical_threshold=0.20):
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
