import time
import os
import cv2

class ViolationAgent:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.evidence_dir = os.path.join(self.output_dir, "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)
        
        self.violations = []
        self.last_violation = {}
        self.cooldown = 5

    def log_violation(self, vtype, frame, detail=""):
        now = time.time()
        if vtype in self.last_violation and now - self.last_violation[vtype] < self.cooldown:
            return

        self.last_violation[vtype] = now
        ts = time.strftime("%H_%M_%S")

        path = os.path.join(self.evidence_dir, f"{vtype}_{ts}.jpg")
        cv2.imwrite(path, frame)

        self.violations.append({
            "time": ts,
            "type": vtype,
            "detail": detail,
            "evidence": path
        })

        print(f"[VIOLATION] {vtype}")