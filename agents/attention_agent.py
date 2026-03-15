import cv2
import numpy as np
import mediapipe as mp
def analyze_attention(self, frame, camera_type="laptop"):

    # Mobile camera cannot reliably detect gaze/iris
    if camera_type == "mobile":
        return {
            "attention": 100,
            "drowsy": False,
            "gaze": "UNKNOWN",
            "head_turn": False,
            "head_turn_severity": "NONE",
            "mouth_open": False,
        }

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

    # ── Yaw smoothing ─────────────────────────────────
    self.yaw_history.append(yaw)
    if len(self.yaw_history) > self.smooth_window:
        self.yaw_history.pop(0)
    yaw = sum(self.yaw_history) / len(self.yaw_history)

    # ── EAR calibration ───────────────────────────────
    if self.calib_frames < self.calib_time:
        self.baseline_ear += ear
        self.calib_frames += 1
        return {
            "attention": 100,
            "drowsy": False,
            "gaze": gaze,
            "head_turn": False,
            "head_turn_severity": "NONE",
            "mouth_open": False,
        }

    if self.calib_frames == self.calib_time:
        self.baseline_ear /= self.calib_time
        self.calib_frames += 1

    # ── Yaw baseline calibration ──────────────────────
    if self.yaw_calib_frames < self.yaw_calib_time:
        self.baseline_yaw += yaw
        self.yaw_calib_frames += 1
        return {
            "attention": 100,
            "drowsy": False,
            "gaze": gaze,
            "head_turn": False,
            "head_turn_severity": "NONE",
            "mouth_open": False,
        }

    if self.yaw_calib_frames == self.yaw_calib_time:
        self.baseline_yaw /= self.yaw_calib_time
        self.yaw_calib_frames += 1

    score = 100

    # ── Drowsiness penalty ────────────────────────────
    if ear < self.baseline_ear * 0.6:
        score -= 40

    # ── Gaze penalty ──────────────────────────────────
    if gaze in ["LEFT_EXTREME", "RIGHT_EXTREME"]:
        score -= 30
    elif gaze in ["LEFT_SOFT", "RIGHT_SOFT"]:
        score -= 5

    # ── Mouth open detection ──────────────────────────
    mouth_open = mar > self.mar_threshold
    if mouth_open:
        score -= 15
        cv2.putText(frame, "MOUTH OPEN", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # ── Head turn detection ───────────────────────────
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

    # ── Pitch detection ───────────────────────────────
    if abs(pitch) > self.head_pitch_limit:
        score -= 15

    # ── Render indicators ─────────────────────────────
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