import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from ultralytics import YOLO


class VisionAgent:

    def __init__(self, model_path: str = "models/yolov8s.pt"):
        self.illegal_objects = ["cell phone", "book", "laptop"]

        # YOLOv8 for object + person detection
        self.model = YOLO(model_path)

        # MediaPipe Tasks API — face detection
        face_model = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "blaze_face_short_range.tflite",
        )
        options = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=face_model),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)

        # Multi-person sustained timer
        self.multi_person_start     = None
        self.multi_person_threshold = 2   # seconds before flagging

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def analyze_vision(self, frame) -> dict:
        """
        Run object detection and face presence check on one BGR frame.

        Returns:
            {
              "illegal_objects": list[str],   # detected prohibited item names
              "multiple_people": bool,         # >1 person sustained >2 s
              "people_count":    int,
              "face_visible":    bool,
            }
        """
        results = self.model(frame, verbose=False)

        people  = 0
        illegal = []

        for r in results:
            for box in r.boxes:
                cls   = int(box.cls)
                label = self.model.names[cls]
                conf  = float(box.conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area < 5000:
                    continue

                if label == "person" and conf > 0.6:
                    people += 1

                if label in self.illegal_objects and conf > 0.45:
                    illegal.append(label)

        # ── Multi-person sustained timer ──────────────────────────
        multi_flag = False

        if people > 1:
            if self.multi_person_start is None:
                self.multi_person_start = time.time()
            elif time.time() - self.multi_person_start > self.multi_person_threshold:
                multi_flag = True
                self.multi_person_start = None
        else:
            self.multi_person_start = None

        # ── Frontal face detection (Tasks API) ────────────────────
        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        face_results = self.face_detector.detect(mp_image)
        face_visible = len(face_results.detections) > 0

        return {
            "illegal_objects": illegal,
            "multiple_people": multi_flag,
            "people_count":    people,
            "face_visible":    face_visible,
        }