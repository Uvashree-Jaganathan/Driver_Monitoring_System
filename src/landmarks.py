import cv2
import mediapipe as mp

class FaceLandmarks:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        landmarks = []

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            for lm in face.landmark:
                landmarks.append((lm.x, lm.y))

        return landmarks, results