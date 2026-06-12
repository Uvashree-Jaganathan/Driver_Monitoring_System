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
