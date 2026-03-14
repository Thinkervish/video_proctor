import time
import os
import cv2
from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["ai_proctoring"]
violations_collection = db["violations"]


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

        # cooldown check
        if vtype in self.last_violation and now - self.last_violation[vtype] < self.cooldown:
            return

        self.last_violation[vtype] = now
        ts = time.strftime("%H_%M_%S")

        # save evidence image
        path = os.path.join(self.evidence_dir, f"{vtype}_{ts}.jpg")
        cv2.imwrite(path, frame)

        violation_data = {
            "time": ts,
            "type": vtype,
            "detail": detail,
            "evidence_path": path,
            "timestamp": time.time()
        }

        # store locally
        self.violations.append(violation_data)

        # store in MongoDB
        try:
            violations_collection.insert_one(violation_data)
        except Exception as e:
            print("MongoDB insert error:", e)