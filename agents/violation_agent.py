import time
import os
import cv2
import numpy as np
from Connections.ViolationLogsDB import violation_logs_collection
import state
from Connections.EvidanceImage import cloudinary

class ViolationAgent:
    def __init__(self, output_dir="outputs"):
        self.violations = []
        self.last_violation = {}
        self.cooldown = 5

    def log_violation(self, vtype, frame, extra=None):
        now = time.time()

        # cooldown check
        if vtype in self.last_violation and now - self.last_violation[vtype] < self.cooldown:
            return

        if frame is None:
            print(f"Skipping {vtype} — frame is None")
            return

        self.last_violation[vtype] = now
        ts = time.strftime("%H_%M_%S") + f"_{int(time.time() * 1000) % 1000:03d}"

        # prepare Cloudinary path
        safe_email = state.Email_id.replace("@", "%40")
        cloud_path = f"{state.Assessment_id}/{safe_email}/{vtype}_{ts}"

        # encode frame to buffer and upload directly to Cloudinary
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            upload_result = cloudinary.uploader.upload(
                buffer.tobytes(),
                public_id=cloud_path,
                resource_type="image"
            )
            cloud_url = upload_result.get("secure_url")
            print(f"Uploaded to Cloudinary: {cloud_url}")
        except Exception as e:
            print("Cloudinary upload error:", e)
            cloud_url = None

        violation_data = {
            "assessment_id": state.Assessment_id,
            "email": state.Email_id,
            "time": ts,
            "type": vtype,
            "cloud_url": cloud_url,
            "timestamp": time.time()
        }

        # store locally in memory
        self.violations.append(violation_data)

        # store in MongoDB
        try:
            print("Logging violation to MongoDB:", violation_data)
            violation_logs_collection.insert_one(violation_data)
        except Exception as e:
            print("MongoDB insert error:", e)