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
