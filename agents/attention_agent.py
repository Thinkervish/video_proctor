"""
agents/attention_agent.py
─────────────────────────────────────────────────────────────────────────────
Handles attention analysis for both cameras:

  camera_type="laptop"  →  front-cam logic (gaze, yaw, EAR, MAR)
  camera_type="mobile"  →  side-cam logic  (pitch=looking down, yaw=looking
                            away from screen, MAR, relaxed EAR)

The return dict shape is identical for both branches so SupervisorAgent
and main.py need no changes.  Side-cam adds one extra key: "looking_down".
"""

import cv2
import numpy as np
import mediapipe as mp


class AttentionAgent:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Eye landmark indices
        self.left_eye  = [33,  160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]

        # ── Front-cam smoothing & calibration ────────────────────
        self.yaw_history    = []
        self.smooth_window  = 5

        self.baseline_ear   = 0.0
        self.calib_frames   = 0
        self.calib_time     = 30

        self.baseline_yaw   = 0.0
        self.yaw_calib_frames = 0
        self.yaw_calib_time   = 30

        # ── Front-cam thresholds ─────────────────────────────────
        self.mar_threshold          = 0.6
        self.head_turn_angle        = 15
        self.head_turn_extreme      = 30
        self.head_turn_frames_required = 3
        self.head_turn_counter      = 0
        self.head_pitch_limit       = 20

        # ── Side-cam smoothing & calibration ─────────────────────
        self.side_yaw_history   = []
        self.side_pitch_history = []

        self.side_baseline_ear        = 0.0
        self.side_ear_calib_frames    = 0
        self.side_ear_calib_time      = 30

        self.side_baseline_yaw        = 0.0
        self.side_baseline_pitch      = 0.0
        self.side_pose_calib_frames   = 0
        self.side_pose_calib_time     = 30

        # ── Side-cam thresholds ──────────────────────────────────
        # Pitch DOWN = looking at lap / phone / notes
        self.side_pitch_down_threshold = 18   # degrees below baseline
        self.side_pitch_down_extreme   = 30
        self.side_pitch_up_threshold   = 20   # looking up = suspicious

        # Yaw from profile = face turning away from screen
        self.side_yaw_away_threshold   = 20
        self.side_yaw_away_extreme     = 35

        # EAR is noisier from a profile — use a more lenient ratio
        self.side_ear_drowsy_ratio     = 0.55

        self.side_mar_threshold        = 0.55

        # Persistence filters
        self.side_head_turn_counter          = 0
        self.side_head_turn_frames_required  = 3
        self.side_pitch_down_counter         = 0
        self.side_pitch_down_frames_required = 3

    # ─────────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────────
    def _head_pose(self, lm, w, h):
        model_points = np.array([
            (0.0,    0.0,    0.0),
            (0.0,  -330.0,  -65.0),
            (-225.0, 170.0, -135.0),
            (225.0,  170.0, -135.0),
            (-150.0,-150.0, -125.0),
            (150.0, -150.0, -125.0),
        ], dtype=np.float64)

        indices      = [1, 152, 33, 263, 61, 291]
        image_points = np.array(
            [(lm[i].x * w, lm[i].y * h) for i in indices], dtype=np.float64
        )
        focal_length = w
        cam_matrix   = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        _, rvec, tvec = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        yaw   = angles[1]
        pitch = angles[0]
        nose  = (int(lm[1].x * w), int(lm[1].y * h))
        return yaw, pitch, nose

    def _get_gaze(self, lm):
        left_iris_x  = lm[468].x
        right_iris_x = lm[473].x
        left_eye_left   = lm[33].x;  left_eye_right  = lm[133].x
        right_eye_left  = lm[362].x; right_eye_right = lm[263].x

        left_ratio  = (left_iris_x  - left_eye_left)  / (left_eye_right  - left_eye_left  + 1e-6)
        right_ratio = (right_iris_x - right_eye_left) / (right_eye_right - right_eye_left + 1e-6)
        avg = (left_ratio + right_ratio) / 2

        if avg < 0.35:   return "LEFT_EXTREME"
        if avg < 0.42:   return "LEFT_SOFT"
        if avg > 0.65:   return "RIGHT_EXTREME"
        if avg > 0.58:   return "RIGHT_SOFT"
        return "CENTER"

    def _get_ear(self, lm, eye_indices, w, h):
        pts = [(lm[i].x * w, lm[i].y * h) for i in eye_indices]
        def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))
        v1  = dist(pts[1], pts[5])
        v2  = dist(pts[2], pts[4])
        hor = dist(pts[0], pts[3])
        return (v1 + v2) / (2.0 * hor + 1e-6)

    def _get_mar(self, lm, w, h):
        top   = (lm[13].x * w,  lm[13].y * h)
        bot   = (lm[14].x * w,  lm[14].y * h)
        left  = (lm[61].x * w,  lm[61].y * h)
        right = (lm[291].x * w, lm[291].y * h)
        v  = np.linalg.norm(np.array(top)  - np.array(bot))
        h_ = np.linalg.norm(np.array(left) - np.array(right))
        return v / (h_ + 1e-6)

    # ─────────────────────────────────────────────────────────────
    # Shared return shape — both branches return this dict.
    # Side-cam adds "looking_down" + "looking_down_severity".
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _base_result(**overrides):
        base = {
            "attention":             100,
            "drowsy":                False,
            "gaze":                  "UNKNOWN",
            "head_turn":             False,
            "head_turn_severity":    "NONE",
            "mouth_open":            False,
            "looking_down":          False,
            "looking_down_severity": "NONE",
        }
        base.update(overrides)
        return base

    # ─────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────
    def analyze(self, frame, camera_type="laptop"):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = self.face_mesh.process(rgb)

        if not mesh.multi_face_landmarks:
            return self._base_result(attention=0)

        lm     = mesh.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # ── Route to the correct branch ───────────────────────────
        if camera_type == "mobile":
            return self._analyze_side(frame, lm, w, h)
        else:
            return self._analyze_front(frame, lm, w, h)

    # ─────────────────────────────────────────────────────────────
    # FRONT-CAM branch  (your original logic, unchanged)
    # ─────────────────────────────────────────────────────────────
    def _analyze_front(self, frame, lm, w, h):
        yaw, pitch, nose = self._head_pose(lm, w, h)
        gaze = self._get_gaze(lm)
        ear  = (self._get_ear(lm, self.left_eye,  w, h) +
                self._get_ear(lm, self.right_eye, w, h)) / 2
        mar  = self._get_mar(lm, w, h)

        # Yaw smoothing
        self.yaw_history.append(yaw)
        if len(self.yaw_history) > self.smooth_window:
            self.yaw_history.pop(0)
        yaw = sum(self.yaw_history) / len(self.yaw_history)

        # EAR calibration
        if self.calib_frames < self.calib_time:
            self.baseline_ear += ear
            self.calib_frames += 1
            return self._base_result(gaze=gaze)
        if self.calib_frames == self.calib_time:
            self.baseline_ear /= self.calib_time
            self.calib_frames += 1

        # Yaw calibration
        if self.yaw_calib_frames < self.yaw_calib_time:
            self.baseline_yaw += yaw
            self.yaw_calib_frames += 1
            return self._base_result(gaze=gaze)
        if self.yaw_calib_frames == self.yaw_calib_time:
            self.baseline_yaw /= self.yaw_calib_time
            self.yaw_calib_frames += 1

        score = 100

        # Drowsiness
        if ear < self.baseline_ear * 0.6:
            score -= 40

        # Gaze penalty
        if gaze in ["LEFT_EXTREME", "RIGHT_EXTREME"]:
            score -= 30
        elif gaze in ["LEFT_SOFT", "RIGHT_SOFT"]:
            score -= 5

        # Mouth open
        mouth_open = mar > self.mar_threshold
        if mouth_open:
            score -= 15
            cv2.putText(frame, "MOUTH OPEN",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Head turn (yaw)
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
                cv2.putText(frame, "HEAD: EXTREME TURN",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                head_turn_severity = "HARD"
                score -= 15
                cv2.putText(frame, "HEAD: HARD TURN",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # Pitch penalty
        if abs(pitch) > self.head_pitch_limit:
            score -= 15

        # Render
        cv2.circle(frame, nose, 3, (0, 255, 255), -1)
        if gaze != "CENTER":
            cv2.putText(frame, f"Gaze: {gaze}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return self._base_result(
            attention          = max(score, 0),
            drowsy             = score < 50,
            gaze               = gaze,
            head_turn          = head_turn,
            head_turn_severity = head_turn_severity,
            mouth_open         = mouth_open,
        )

    # ─────────────────────────────────────────────────────────────
    # SIDE-CAM branch  (90° profile, pitch = primary signal)
    # ─────────────────────────────────────────────────────────────
    def _analyze_side(self, frame, lm, w, h):
        yaw, pitch, _ = self._head_pose(lm, w, h)
        ear = (self._get_ear(lm, self.left_eye,  w, h) +
               self._get_ear(lm, self.right_eye, w, h)) / 2
        mar = self._get_mar(lm, w, h)

        # Smoothing
        self.side_yaw_history.append(yaw)
        self.side_pitch_history.append(pitch)
        if len(self.side_yaw_history)   > self.smooth_window: self.side_yaw_history.pop(0)
        if len(self.side_pitch_history) > self.smooth_window: self.side_pitch_history.pop(0)
        yaw   = sum(self.side_yaw_history)   / len(self.side_yaw_history)
        pitch = sum(self.side_pitch_history) / len(self.side_pitch_history)

        # EAR calibration
        if self.side_ear_calib_frames < self.side_ear_calib_time:
            self.side_baseline_ear += ear
            self.side_ear_calib_frames += 1
            return self._base_result(gaze="CALIBRATING")
        if self.side_ear_calib_frames == self.side_ear_calib_time:
            self.side_baseline_ear /= self.side_ear_calib_time
            self.side_ear_calib_frames += 1

        # Pose calibration
        if self.side_pose_calib_frames < self.side_pose_calib_time:
            self.side_baseline_yaw   += yaw
            self.side_baseline_pitch += pitch
            self.side_pose_calib_frames += 1
            return self._base_result(gaze="CALIBRATING")
        if self.side_pose_calib_frames == self.side_pose_calib_time:
            self.side_baseline_yaw   /= self.side_pose_calib_time
            self.side_baseline_pitch /= self.side_pose_calib_time
            self.side_pose_calib_frames += 1

        score = 100

        # 1. Drowsiness — relaxed EAR ratio for profile view
        drowsy = ear < self.side_baseline_ear * self.side_ear_drowsy_ratio
        if drowsy:
            score -= 35

        # 2. Mouth open
        mouth_open = mar > self.side_mar_threshold
        if mouth_open:
            score -= 15
            cv2.putText(frame, "SIDE: MOUTH OPEN",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 3. Looking DOWN — pitch below baseline (lap / phone / notes)
        pitch_diff            = self.side_baseline_pitch - pitch
        looking_down          = False
        looking_down_severity = "NONE"

        if pitch_diff > self.side_pitch_down_threshold:
            self.side_pitch_down_counter += 1
        else:
            self.side_pitch_down_counter = max(0, self.side_pitch_down_counter - 1)

        if self.side_pitch_down_counter >= self.side_pitch_down_frames_required:
            looking_down = True
            if pitch_diff > self.side_pitch_down_extreme:
                looking_down_severity = "HARD"
                score -= 30
                cv2.putText(frame, "SIDE: LOOKING DOWN (EXTREME)",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                looking_down_severity = "SOFT"
                score -= 15
                cv2.putText(frame, "SIDE: LOOKING DOWN",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # 4. Looking AWAY from screen — yaw off-baseline from profile
        yaw_diff           = abs(self.side_baseline_yaw - yaw)
        head_turn          = False
        head_turn_severity = "NONE"

        if yaw_diff > self.side_yaw_away_threshold:
            self.side_head_turn_counter += 1
        else:
            self.side_head_turn_counter = max(0, self.side_head_turn_counter - 1)

        if self.side_head_turn_counter >= self.side_head_turn_frames_required:
            head_turn = True
            if yaw_diff > self.side_yaw_away_extreme:
                head_turn_severity = "HARD"
                score -= 25
                cv2.putText(frame, "SIDE: HEAD TURN (EXTREME)",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                head_turn_severity = "HARD"
                score -= 15
                cv2.putText(frame, "SIDE: HEAD TURN",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        # 5. Looking UP — unusual, flag lightly
        if (pitch - self.side_baseline_pitch) > self.side_pitch_up_threshold:
            score -= 10

        return self._base_result(
            attention             = max(score, 0),
            drowsy                = drowsy,
            gaze                  = "PROFILE",   # iris ratio unreliable at 90°
            head_turn             = head_turn,
            head_turn_severity    = head_turn_severity,
            mouth_open            = mouth_open,
            looking_down          = looking_down,
            looking_down_severity = looking_down_severity,
        )