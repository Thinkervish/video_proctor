import time
import cv2
import mediapipe as mp
from ultralytics import YOLO


class VisionAgent:

    def __init__(self, model_path: str = "models/yolov8s.pt"):
        self.illegal_objects = ["cell phone", "book", "laptop"]

        # YOLOv8 for object + person detection
        self.model = YOLO(model_path)

        # MediaPipe frontal face detection
        mp_face = mp.solutions.face_detection
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

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

        # ── Frontal face detection ─────────────────────────────────
        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.process(rgb_frame)
        face_visible = bool(face_results.detections)

        return {
            "illegal_objects": illegal,
            "multiple_people": multi_flag,
            "people_count":    people,
            "face_visible":    face_visible,
        }