import time
import os
import cv2
from pymongo import MongoClient
from  Connections.ViolationLogsDB import violation_logs_collection
import state

class ViolationAgent:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.evidence_dir = os.path.join(self.output_dir, "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)

        self.violations = []
        self.last_violation = {}
        self.cooldown = 5

    def log_violation(self, vtype, frame ):
        now = time.time()

        # cooldown check
        if vtype in self.last_violation and now - self.last_violation[vtype] < self.cooldown:
            return

        self.last_violation[vtype] = now
        ts = time.strftime("%H_%M_%S")

        # save evidence image
        path = os.path.join(self.evidence_dir, f"{vtype}_{ts}.jpg")
        cv2.imwrite(path, frame)

        violation_data = {
            "assessment_id": state.Assessment_id,
            "email": state.Email_id,
            "time": ts,
            "type": vtype,
            "evidence_path": path,
            "timestamp": time.time()
        }

        # store locally
        self.violations.append(violation_data)

        # store in MongoDB
        try:
            print("Logging violation to MongoDB:", violation_data)
            violation_logs_collection.insert_one(violation_data)
        except Exception as e:
            print("MongoDB insert error:", e)