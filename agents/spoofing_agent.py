import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


class SpoofingAgent:

    def __init__(self):
        # MediaPipe Tasks API — face landmarker (replaces FaceMesh)
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "face_landmarker.task",
        )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

        # Eye landmark indices for blink detection
        self.left_eye  = [33,  160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]

        # ── Texture analysis ──────────────────────────────────────
        self.texture_variance_threshold = 500.0
        self.spoof_confidence_threshold = 0.65

        # ── Score smoothing ───────────────────────────────────────
        self.score_history = []
        self.history_size  = 15

        # ── Blink liveness tracker ────────────────────────────────
        self.ear_history        = []
        self.blink_detected     = False
        self.frames_since_blink = 0
        self.no_blink_limit     = 150   # ~5 s at 30 fps → suspicious if no blink

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _get_ear(self, lm, eye_ids, w, h) -> float:
        """Eye Aspect Ratio — lower = more closed."""
        pts   = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_ids]
        v1    = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2    = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * h_dist + 1e-6)

    def _lbp_variance(self, roi_gray) -> float:
        """
        Sobel-gradient variance on the face ROI.
        Real skin → high variance.  Printed / screen face → low variance.
        """
        if roi_gray is None or roi_gray.size == 0:
            return 9999.0   # can't measure → assume real

        roi    = cv2.resize(roi_gray, (64, 64))
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return float(np.var(magnitude))

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def analyze_spoofing(self, frame) -> dict:
        """
        Analyse one BGR frame for liveness.

        Returns:
            {
              "is_spoof":   bool,
              "confidence": float,   # 0.0 – 1.0
              "reason":     str,     # real_face / flat_texture /
                                     # no_blink_detected /
                                     # flat_texture_and_no_blink
            }
        """
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_mesh.detect(mp_image)

        if not result.face_landmarks:
            return {"is_spoof": False, "confidence": 0.0, "reason": "no_face"}

        lm = result.face_landmarks[0]

        # ── Signal 1: Texture variance ────────────────────────────
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))

        face_roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.size > 0 else None

        texture_var   = self._lbp_variance(gray_roi)
        texture_score = 1.0 if texture_var < self.texture_variance_threshold else 0.0

        # ── Signal 2: Blink liveness ──────────────────────────────
        ear = (
            self._get_ear(lm, self.left_eye,  w, h) +
            self._get_ear(lm, self.right_eye, w, h)
        ) / 2.0

        self.ear_history.append(ear)
        if len(self.ear_history) > 5:
            self.ear_history.pop(0)

        # Blink = EAR drops below 0.18 then rises above 0.20
        if len(self.ear_history) >= 3:
            if self.ear_history[-2] < 0.18 and self.ear_history[-1] > 0.20:
                self.blink_detected     = True
                self.frames_since_blink = 0

        if not self.blink_detected:
            self.frames_since_blink += 1

        no_blink_score = 1.0 if self.frames_since_blink > self.no_blink_limit else 0.0

        # ── Combine signals (60/40 split) ─────────────────────────
        spoof_score = (texture_score * 0.60) + (no_blink_score * 0.40)

        self.score_history.append(spoof_score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)

        smoothed = float(np.mean(self.score_history))
        is_spoof = smoothed >= self.spoof_confidence_threshold

        if is_spoof:
            if texture_score > 0 and no_blink_score > 0:
                reason = "flat_texture_and_no_blink"
            elif texture_score > 0:
                reason = "flat_texture"
            else:
                reason = "no_blink_detected"
        else:
            reason = "real_face"

        # ── Frame overlay ─────────────────────────────────────────
        color = (0, 0, 255) if is_spoof else (0, 255, 0)
        label = (
            f"SPOOF ({smoothed:.2f})" if is_spoof
            else f"REAL ({1 - smoothed:.2f})"
        )
        cv2.putText(
            frame, label,
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )
        if is_spoof:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        return {
            "is_spoof":   is_spoof,
            "confidence": round(smoothed, 3),
            "reason":     reason,
        }