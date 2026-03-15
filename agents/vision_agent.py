import time
from ultralytics import YOLO
import mediapipe as mp
import cv2


class VisionAgent:

    def __init__(self, model_path="models/yolov8s.pt"):

        self.illegal_objects = ["cell phone", "book", "laptop"]

        self.model = YOLO(model_path)

        mp_face = mp.solutions.face_detection
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

        self.multi_person_start = None
        self.multi_person_threshold = 2


    def analyze_vision(self, frame, camera_type="laptop"):

        results = self.model(frame, verbose=False)

        people = 0
        illegal = []

        for r in results:
            for box in r.boxes:

                cls = int(box.cls)
                label = self.model.names[cls]
                conf = float(box.conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area < 5000:
                    continue

                if label == "person" and conf > 0.6:
                    people += 1

                if label in self.illegal_objects and conf > 0.45:
                    illegal.append(label)

        # Multi-person logic
        multi_flag = False

        if people > 1:

            if self.multi_person_start is None:
                self.multi_person_start = time.time()

            elif time.time() - self.multi_person_start > self.multi_person_threshold:
                multi_flag = True
                self.multi_person_start = None

        else:
            self.multi_person_start = None


        # ------------------------------
        # Face detection (Laptop only)
        # ------------------------------

        face_visible = True

        if camera_type == "laptop":

            face_results = self.face_detector.process(frame)
            face_visible = bool(face_results.detections)

        else:
            # Mobile camera may see side of face or body
            face_visible = True


        return {
            "camera": camera_type,
            "illegal_objects": illegal,
            "multiple_people": multi_flag,
            "people_count": people,
            "face_visible": face_visible
        }