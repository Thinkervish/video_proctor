import time
import os
import cv2
from Connections.ViolationLogsDB import violation_logs_collection
import state
import cloudinary.uploader
from Connections.EvidanceImage import cloudinary

class ViolationAgent:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.evidence_dir = os.path.join(self.output_dir, "evidence")
        os.makedirs(self.evidence_dir, exist_ok=True)

        self.violations = []
        self.last_violation = {}
        self.cooldown = 5

    def log_violation(self, vtype, frame, extra=None):
        now = time.time()

        # cooldown check
        if vtype in self.last_violation and now - self.last_violation[vtype] < self.cooldown:
            return

        self.last_violation[vtype] = now
        ts = time.strftime("%H_%M_%S")

        # save local evidence image
        local_path = os.path.join(self.evidence_dir, f"{vtype}_{ts}.jpg")
      

        # prepare Cloudinary path
        safe_email = state.Email_id.replace("@", "%40")  # convert email for safe path
        cloud_path = f"{state.Assessment_id}/{safe_email}/{vtype}_{ts}"

        # upload to Cloudinary
        try:
            upload_result = cloudinary.uploader.upload(local_path, public_id=cloud_path)
            cloud_url = upload_result.get("secure_url")
        except Exception as e:
            print("Cloudinary upload error:", e)
            cloud_url = None

        violation_data = {
            "assessment_id": state.Assessment_id,
            "email": state.Email_id,
            "time": ts,
            "type": vtype,
            "evidence_path": local_path,
            "cloud_url": cloud_url,
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