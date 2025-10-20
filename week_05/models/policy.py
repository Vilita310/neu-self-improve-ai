
"""Lightweight model wrapper."""

class DummyPolicy:
    def __init__(self):
        print("[INFO] Initialized dummy policy model (for CPU test).")
    def generate(self, prompt):
        return "x = 1"
