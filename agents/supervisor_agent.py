"""
agents/supervisor_agent.py  (updated — side cam aware)
─────────────────────────────────────────────────────────────────────────────
Routes violations from both front cam (camera_type="laptop") and side cam
(camera_type="mobile") to the shared RiskAgent and ViolationAgent.

Side-cam-specific additions:
  • looking_down  → candidate looking at lap / phone / notes
  • head_turn     → candidate looking away from screen (detected from profile)
  • Attention threshold lowered for side cam (profile EAR is noisier)
"""


class SupervisorAgent:
    def __init__(self, risk_agent, violation_agent):
        self.risk_agent      = risk_agent
        self.violation_agent = violation_agent

    # ─────────────────────────────────────────────────────────────────────
    def supervise(
        self,
        vision_data,
        attention_data,
        audio_data,
        frame,
        camera_type="laptop",
        spoof_data=None,
    ):
        is_side = camera_type == "mobile"

        # ── Vision checks ─────────────────────────────────────────────────
        if not vision_data["face_visible"]:
            self.violation_agent.log_violation("face_not_visible", frame, camera_type)
            self.risk_agent.update_risk("face_not_visible")

        if vision_data["multiple_people"]:
            self.violation_agent.log_violation("multiple_people", frame, camera_type)
            self.risk_agent.update_risk("multiple_people")

        for obj in vision_data["illegal_objects"]:
            self.violation_agent.log_violation("illegal_object", frame, obj)
            self.risk_agent.update_risk("illegal_object")

        # ── Attention checks ──────────────────────────────────────────────
        # Side cam: use a slightly higher leniency threshold (profile EAR noisier)
        attention_threshold = 25 if is_side else 30

        if attention_data["attention"] < attention_threshold:
            self.violation_agent.log_violation("low_attention", frame, camera_type)
            self.risk_agent.update_risk("low_attention")

        if attention_data["drowsy"]:
            self.violation_agent.log_violation("drowsy", frame, camera_type)
            self.risk_agent.update_risk("drowsy")

        if attention_data.get("head_turn", False):
            self.violation_agent.log_violation("head_turned", frame, camera_type)
            self.risk_agent.update_risk("head_turned")

        if attention_data.get("mouth_open", False):
            self.violation_agent.log_violation("mouth_open", frame, camera_type)
            self.risk_agent.update_risk("mouth_open")

        # ── Side-cam-only: looking DOWN ───────────────────────────────────
        # (front cam cannot detect this reliably; side cam is purpose-built for it)
        if is_side and attention_data.get("looking_down", False):
            self.violation_agent.log_violation("looking_down", frame, camera_type)
            self.risk_agent.update_risk("looking_down")

        # ── Audio checks ──────────────────────────────────────────────────
        # Audio is processed once globally (AudioAgent runs independently),
        # so we only apply audio checks on the front-cam pass to avoid
        # double-counting the same audio event.
        if not is_side:
            if audio_data.get("talking", False):
                self.violation_agent.log_violation("talking", frame, camera_type)
                self.risk_agent.update_risk("talking")

            if audio_data.get("loud_noise", False):
                self.violation_agent.log_violation("loud_noise", frame, camera_type)
                self.risk_agent.update_risk("loud_noise")

        # ── Spoofing check ────────────────────────────────────────────────
        if spoof_data and spoof_data.get("is_spoof", False):
            self.violation_agent.log_violation("spoofing_attempt", frame, camera_type)
            self.risk_agent.update_risk("spoofing_attempt")