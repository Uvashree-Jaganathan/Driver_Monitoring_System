import numpy as np

# We define these here so they are available to any file that imports utils
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def get_eye_points(landmarks_list, eye_indices):
    """
    Extracts specific eye points from the full facial landmark list.
    landmarks_list: List of (x, y) coordinates
    eye_indices: List of integers (LEFT_EYE_INDICES or RIGHT_EYE_INDICES)
    """
    # Convert the selected landmarks into a numpy array for math operations
    return np.array([landmarks_list[idx] for idx in eye_indices])

def calculate_ear(eye_points):
    """
    Computes the Eye Aspect Ratio (EAR).
    """
    # Vertical distances
    # np.linalg.norm calculates the Euclidean distance between two points
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    
    # Horizontal distance
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    
    # The EAR formula
    ear = (A + B) / (2.0 * C)
    return ear