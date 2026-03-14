import cv2
import numpy as np
import mediapipe as mp


class SpoofingAgent:
    def __init__(self):
        # =====================================================================
        # MediaPipe Face Mesh for fine landmarks (blink, texture analysis)
        # =====================================================================
        mp_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye indices for blink detection (liveness cue)
        self.left_eye  = [33,  160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]

        # =====================================================================
        # LBP Texture Analysis thresholds
        # Real faces have HIGH texture variance (skin pores, shadows, 3D depth).
        # Printed / screen photos are flat → LOW texture variance.
        # =====================================================================
        self.texture_variance_threshold = 500.0   # Below this → possible spoof
        self.spoof_confidence_threshold = 0.65    # Above this confidence → flag as spoof

        # Frame history for temporal smoothing (avoids single-frame false alarms)
        self.score_history = []
        self.history_size  = 15   # Smooth over last 15 frames

        # Blink tracker — real humans blink. A photo never blinks.
        self.ear_history   = []
        self.blink_detected = False
        self.frames_since_blink = 0
        self.no_blink_limit = 150   # ~5 seconds at 30fps → flag if no blink

    # =========================================================================
    # PRIVATE: Eye Aspect Ratio (for blink liveness check)
    # =========================================================================
    def _get_ear(self, lm, eye_ids, w, h):
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_ids]
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * h_dist + 1e-6)

    # =========================================================================
    # PRIVATE: LBP Texture Variance (real skin vs printed/screen)
    # =========================================================================
    def _lbp_variance(self, roi_gray):
        if roi_gray is None or roi_gray.size == 0:
            return 9999.0   # Unknown — assume real (fail safe)

        # Resize for consistency
        roi = cv2.resize(roi_gray, (64, 64))

        # LBP-like: compute gradient magnitude as texture proxy
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.var(magnitude))

    # =========================================================================
    # PUBLIC: MAIN ANALYSIS METHOD
    # =========================================================================
    def analyze_spoofing(self, frame):
        """
        Analyzes the frame for spoofing indicators.
        Returns:
            {
                "is_spoof":   bool,
                "confidence": float (0.0 – 1.0),
                "reason":     str
            }
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        # No face → can't assess → return safe
        if not result.multi_face_landmarks:
            return {"is_spoof": False, "confidence": 0.0, "reason": "no_face"}

        lm = result.multi_face_landmarks[0].landmark

        # =============================================================
        # CUE 1: Texture Analysis on face ROI
        # =============================================================
        # Get face bounding box from landmarks
        xs = [int(p.x * w) for p in lm]
        ys = [int(p.y * h) for p in lm]
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))

        face_roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.size > 0 else None
        texture_var = self._lbp_variance(gray_roi)

        # Low texture variance → more likely spoof
        texture_score = 1.0 if texture_var < self.texture_variance_threshold else 0.0

        # =============================================================
        # CUE 2: Blink Liveness Check
        # =============================================================
        ear_l = self._get_ear(lm, self.left_eye, w, h)
        ear_r = self._get_ear(lm, self.right_eye, w, h)
        ear   = (ear_l + ear_r) / 2.0

        self.ear_history.append(ear)
        if len(self.ear_history) > 5:
            self.ear_history.pop(0)

        # Blink: EAR drops sharply then recovers
        if len(self.ear_history) >= 3:
            if self.ear_history[-2] < 0.18 and self.ear_history[-1] > 0.20:
                self.blink_detected = True
                self.frames_since_blink = 0

        if not self.blink_detected:
            self.frames_since_blink += 1

        # Flag no-blink as suspicious after limit
        no_blink_score = 1.0 if self.frames_since_blink > self.no_blink_limit else 0.0

        # =============================================================
        # COMBINE CUES → Spoof Score
        # Weighted: texture (60%) + blink (40%)
        # =============================================================
        spoof_score = (texture_score * 0.60) + (no_blink_score * 0.40)

        # Smooth score over history to prevent jitter
        self.score_history.append(spoof_score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)
        smoothed_score = float(np.mean(self.score_history))

        is_spoof = smoothed_score >= self.spoof_confidence_threshold

        # Determine human-readable reason
        if is_spoof:
            if texture_score > 0 and no_blink_score > 0:
                reason = "flat_texture_and_no_blink"
            elif texture_score > 0:
                reason = "flat_texture"
            else:
                reason = "no_blink_detected"
        else:
            reason = "real_face"

        # =============================================================
        # Draw spoof indicator on frame (in-place)
        # =============================================================
        color = (0, 0, 255) if is_spoof else (0, 255, 0)
        label = f"SPOOF ({smoothed_score:.2f})" if is_spoof else f"REAL ({1 - smoothed_score:.2f})"
        cv2.putText(frame, label, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        if is_spoof:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        return {
            "is_spoof":   is_spoof,
            "confidence": round(smoothed_score, 3),
            "reason":     reason
        }
