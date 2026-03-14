class SupervisorAgent:
    def __init__(self, risk_agent, violation_agent):
        self.risk_agent      = risk_agent
        self.violation_agent = violation_agent

    def supervise(self, vision_data, attention_data, audio_data, frame, spoof_data=None):

        # ── Vision checks ──────────────────────────────────────────────────
        if not vision_data["face_visible"]:
            self.violation_agent.log_violation("face_not_visible", frame)
            self.risk_agent.update_risk("face_not_visible")

        if vision_data["multiple_people"]:
            self.violation_agent.log_violation("multiple_people", frame)
            self.risk_agent.update_risk("multiple_people")

        for obj in vision_data["illegal_objects"]:
            self.violation_agent.log_violation("illegal_object", frame, obj)
            self.risk_agent.update_risk("illegal_object")

        # ── Attention checks ───────────────────────────────────────────────
        if attention_data["attention"] < 30:
            self.violation_agent.log_violation("low_attention", frame)
            self.risk_agent.update_risk("low_attention")

        if attention_data["drowsy"]:
            self.violation_agent.log_violation("drowsy", frame)
            self.risk_agent.update_risk("drowsy")

        if attention_data.get("head_turn", False):
            self.violation_agent.log_violation("head_turned", frame)
            self.risk_agent.update_risk("head_turned")

        if attention_data.get("mouth_open", False):
            self.violation_agent.log_violation("mouth_open", frame)
            self.risk_agent.update_risk("mouth_open")

        # ── Audio checks ───────────────────────────────────────────────────
        if audio_data.get("talking", False):
            self.violation_agent.log_violation("talking", frame)
            self.risk_agent.update_risk("talking")

        if audio_data.get("loud_noise", False):
            self.violation_agent.log_violation("loud_noise", frame)
            self.risk_agent.update_risk("loud_noise")

        # ── Spoofing check (optional) ──────────────────────────────────────
        if spoof_data and spoof_data.get("is_spoof", False):
            self.violation_agent.log_violation("spoofing_attempt", frame)
            self.risk_agent.update_risk("spoofing_attempt")