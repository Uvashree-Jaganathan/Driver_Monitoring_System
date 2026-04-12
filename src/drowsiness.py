import time
from utils.eye_utils import calculate_ear

class DrowsinessDetector:
    def __init__(self, threshold=0.22, drowsy_time=2.0):
        # The EAR value below which we consider eyes "closed"
        self.threshold = threshold
        # How many seconds the eyes must stay closed to trigger an alert
        self.drowsy_time = drowsy_time
        # Stores the timestamp when eyes first closed
        self.start_time = None

    def check_status(self, left_eye, right_eye):
        """
        Processes eye landmarks and returns the driver status.
        """
        # 1. Calculate the current EAR
        ear_l = calculate_ear(left_eye)
        ear_r = calculate_ear(right_eye)
        avg_ear = (ear_l + ear_r) / 2.0

        # 2. Determine Status based on Timer
        status = "ALERT"
        
        if avg_ear < self.threshold:
            # If this is the first frame the eyes are closed, start the clock
            if self.start_time is None:
                self.start_time = time.time()
            
            # Check if the duration of closure exceeds our limit
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.drowsy_time:
                status = "DROWSY!"
        else:
            # Eyes are open; reset the timer immediately
            self.start_time = None

        return status, avg_ear