import numpy as np
import mediapipe as mp
import cv2


class AttentionAgent:
    def __init__(self):

        mp_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_mesh.FaceMesh(refine_landmarks=True)

        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]
        self.left_iris = 468
        self.right_iris = 473

        # Mouth landmarks for MAR (Mouth Aspect Ratio)
        self.mouth_top    = 13
        self.mouth_bottom = 14
        self.mouth_left   = 78
        self.mouth_right  = 308
        self.mar_threshold = 0.35   # above this → mouth open

        self.baseline_ear = 0
        self.calib_frames = 0
        self.calib_time = 60

        # Yaw baseline calibration
        self.baseline_yaw = 0
        self.yaw_calib_frames = 0
        self.yaw_calib_time = 60

        # Head turn detection thresholds
        self.head_turn_angle   = 5
        self.head_turn_extreme = 15

        self.head_pitch_limit = 250

        # Stability control
        self.head_turn_counter = 0
        self.head_turn_frames_required = 10

        # Yaw smoothing
        self.yaw_history = []
        self.smooth_window = 5


    def _get_ear(self, lm, eye, w, h):
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye]
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * h_dist + 1e-6)


    def _get_mar(self, lm, w, h):
        """
        Mouth Aspect Ratio — detects open mouth (whispering/talking).
        MAR = vertical opening / horizontal width
        Above mar_threshold → mouth open
        """
        top    = np.array([lm[self.mouth_top].x * w,    lm[self.mouth_top].y * h])
        bottom = np.array([lm[self.mouth_bottom].x * w, lm[self.mouth_bottom].y * h])
        left   = np.array([lm[self.mouth_left].x * w,   lm[self.mouth_left].y * h])
        right  = np.array([lm[self.mouth_right].x * w,  lm[self.mouth_right].y * h])

        vertical   = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right) + 1e-6

        return vertical / horizontal


    def _get_gaze(self, lm):
        avg_x = (lm[self.left_iris].x + lm[self.right_iris].x) / 2

        if avg_x < 0.35: return "LEFT_EXTREME"
        if avg_x < 0.45: return "LEFT_SOFT"
        if avg_x > 0.65: return "RIGHT_EXTREME"
        if avg_x > 0.55: return "RIGHT_SOFT"
        return "CENTER"


    def _head_pose(self, lm, w, h):
        ids = [1, 33, 263, 152, 10]
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in ids]
        nose, l, r, chin, head = pts
        yaw   = r[0] - l[0]
        pitch = chin[1] - head[1]
        return yaw, pitch, nose


    def analyze_attention(self, frame):

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = self.face_mesh.process(rgb)

        if not mesh.multi_face_landmarks:
            return {
                "attention":          0,
                "drowsy":             False,
                "gaze":               "UNKNOWN",
                "head_turn":          False,
                "head_turn_severity": "NONE",
                "mouth_open":         False,
            }

        lm = mesh.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        yaw, pitch, nose = self._head_pose(lm, w, h)
        gaze = self._get_gaze(lm)
        ear  = (self._get_ear(lm, self.left_eye, w, h) +
                self._get_ear(lm, self.right_eye, w, h)) / 2
        mar  = self._get_mar(lm, w, h)

        # ── Yaw smoothing ─────────────────────────────────────────────────
        self.yaw_history.append(yaw)
        if len(self.yaw_history) > self.smooth_window:
            self.yaw_history.pop(0)
        yaw = sum(self.yaw_history) / len(self.yaw_history)

        # ── EAR calibration ───────────────────────────────────────────────
        if self.calib_frames < self.calib_time:
            self.baseline_ear += ear
            self.calib_frames += 1
            return {
                "attention": 100, "drowsy": False, "gaze": gaze,
                "head_turn": False, "head_turn_severity": "NONE", "mouth_open": False,
            }

        if self.calib_frames == self.calib_time:
            self.baseline_ear /= self.calib_time
            self.calib_frames += 1

        # ── Yaw baseline calibration ──────────────────────────────────────
        if self.yaw_calib_frames < self.yaw_calib_time:
            self.baseline_yaw += yaw
            self.yaw_calib_frames += 1
            return {
                "attention": 100, "drowsy": False, "gaze": gaze,
                "head_turn": False, "head_turn_severity": "NONE", "mouth_open": False,
            }

        if self.yaw_calib_frames == self.yaw_calib_time:
            self.baseline_yaw /= self.yaw_calib_time
            self.yaw_calib_frames += 1

        score = 100

        # ── Drowsiness penalty ────────────────────────────────────────────
        if ear < self.baseline_ear * 0.6:
            score -= 40

        # ── Gaze penalty ──────────────────────────────────────────────────
        if gaze in ["LEFT_EXTREME", "RIGHT_EXTREME"]:
            score -= 30
        elif gaze in ["LEFT_SOFT", "RIGHT_SOFT"]:
            score -= 5

        # ── Mouth open detection ──────────────────────────────────────────
        mouth_open = mar > self.mar_threshold
        if mouth_open:
            score -= 15
            cv2.putText(frame, "MOUTH OPEN", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # ── Head turn detection ───────────────────────────────────────────
        head_turn          = False
        head_turn_severity = "NONE"

        yaw_diff = self.baseline_yaw - yaw

        if yaw_diff > self.head_turn_angle:
            self.head_turn_counter += 1
        else:
            self.head_turn_counter = max(0, self.head_turn_counter - 1)

        if self.head_turn_counter >= self.head_turn_frames_required:
            head_turn = True

            if yaw_diff > self.head_turn_extreme:
                head_turn_severity = "HARD"
                score -= 25
                cv2.putText(frame, "HEAD: EXTREME TURN", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                head_turn_severity = "HARD"
                score -= 15
                cv2.putText(frame, "HEAD: HARD TURN", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # ── Pitch detection ───────────────────────────────────────────────
        if abs(pitch) > self.head_pitch_limit:
            score -= 15

        # ── Render indicators ─────────────────────────────────────────────
        cv2.circle(frame, nose, 3, (0, 255, 255), -1)
        if gaze != "CENTER":
            cv2.putText(frame, f"Gaze: {gaze}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return {
            "attention":          max(score, 0),
            "drowsy":             score < 50,
            "gaze":               gaze,
            "head_turn":          head_turn,
            "head_turn_severity": head_turn_severity,
            "mouth_open":         mouth_open,
        }