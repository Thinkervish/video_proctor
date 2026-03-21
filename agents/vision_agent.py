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

        # FIX: separate timers per camera so front and side cam
        # don't corrupt each other's multi-person countdown
        self.multi_person_start = {
            "laptop": None,
            "mobile": None,
        }
        self.multi_person_threshold = 2

    def analyze_vision(self, frame, camera_type="laptop"):

        results = self.model(frame, verbose=False)

        people = 0
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

        # ── Multi-person logic (per-camera timer) ────────────────
        multi_flag = False
        timer_key  = camera_type if camera_type in self.multi_person_start else "laptop"

        if people > 1:
            if self.multi_person_start[timer_key] is None:
                self.multi_person_start[timer_key] = time.time()
            elif time.time() - self.multi_person_start[timer_key] > self.multi_person_threshold:
                multi_flag = True
                self.multi_person_start[timer_key] = None
        else:
            self.multi_person_start[timer_key] = None

        # ── Face detection ───────────────────────────────────────
        # Laptop: full frontal detection
        # Mobile: side-profile detection — same detector, lower confidence
        #         since a profile face scores lower than a frontal face
        face_visible = True

        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.process(rgb_frame)

        if camera_type == "laptop":
            face_visible = bool(face_results.detections)

        elif camera_type == "mobile":
            # Profile faces score lower — accept any detection above 0.3
            # (detector was initialised at 0.5 but detections object still
            #  carries the raw score we can check)
            if face_results.detections:
                best_score = max(
                    d.score[0] for d in face_results.detections
                )
                face_visible = best_score >= 0.3
            else:
                face_visible = False

        return {
            "camera":          camera_type,
            "illegal_objects": illegal,
            "multiple_people": multi_flag,
            "people_count":    people,
            "face_visible":    face_visible,
        }