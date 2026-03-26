"""
agents/risk_agent.py
─────────────────────────────────────────────────────────────────────────────
Maintains the running suspicion score, violation timeline, burst detection,
trust cutoff alerts, and tab-switch 2-strike logic.

Laptop / front-cam only — no side-cam events.

Scoring:
  - All violations : flat 5 pts each
  - Max score cap  : 50  (rescaled for distributed trust)
  - Tab switch     : +50 pts (1st → final warning, 2nd → test terminated)
"""

import time
from collections import defaultdict

WARNING_MESSAGES = {
    "illegal_object":   ("📱 Illegal Object Detected",   "Please remove all prohibited items from view immediately."),
    "multiple_people":  ("👥 Multiple People Detected",  "Only the candidate should be visible on camera."),
    "face_not_visible": ("👤 Face Not Visible",          "Please ensure your face is clearly visible on camera."),
    "head_turned":      ("↩️ Head Turn Detected",        "Please keep your eyes on the screen at all times."),
    "talking":          ("🗣️ Talking Detected",          "Please maintain silence during the exam."),
    "loud_noise":       ("🔊 Loud Noise Detected",       "Suspicious audio activity has been flagged."),
    "drowsy":           ("😴 Drowsiness Detected",       "Please stay alert and focused on your exam."),
    "mouth_open":       ("👄 Whispering Suspected",      "Unusual mouth movement detected. Please stay silent."),
    "low_attention":    ("😶 Low Attention",             "Please focus on the exam screen."),
    "spoofing_attempt": ("🎭 Spoofing Attempt Detected", "Identity verification failed. This has been flagged."),
}


class RiskAgent:

    # Maximum suspicion score for video proctoring violations
    MAX_SCORE = 50

    def __init__(self):
        self.suspicion_score = 0
        self.timeline        = []

        # All violations carry a flat 5-point penalty
        self.weights = {
            "illegal_object":   5,
            "multiple_people":  5,
            "low_attention":    0,
            "drowsy":           0,
            "face_not_visible": 5,
            "head_turned":      5,
            "talking":          5,
            "loud_noise":       5,
            "spoofing_attempt": 5,
            "mouth_open":       5,
        }

        self.violation_counts       = defaultdict(int)
        self.last_risk_time         = {}
        self.risk_cooldown          = 5      # seconds between same-event logs

        # ── Tab switch: 2-strike special ─────────────────────────
        self.tab_switch_count = 0
        self.test_terminated  = False

        # ── Active warning ────────────────────────────────────────
        self.active_warning  = None
        self.warning_history = []

        # ── Burst detection ───────────────────────────────────────
        self.burst_window        = 60
        self.burst_threshold     = 4
        self.recent_events       = []
        self.burst_bonus_applied = False

        # ── Trust cutoff alert ────────────────────────────────────
        self.alert_active        = False
        self.alert_triggered_at  = None
        self.alert_messages      = []
        self.cutoff_breach_count = 0

    def update_risk(self, event: str) -> None:
        now = time.time()
        if self.test_terminated: return

        if event == "tab_switched":
            self._handle_tab_switch(now)
            return

        if event in self.last_risk_time:
            if now - self.last_risk_time[event] < self.risk_cooldown: return
        self.last_risk_time[event] = now

        if event not in self.weights: return

        self.violation_counts[event] += 1
        repeat  = self.violation_counts[event]
        penalty = self.weights[event]

        old = self.suspicion_score
        self.suspicion_score = min(self.MAX_SCORE, self.suspicion_score + penalty)

        self.timeline.append({
            "event":   event,
            "score":   self.suspicion_score,
            "time":    time.strftime("%H:%M:%S"),
            "repeat":  repeat,
            "penalty": penalty,
            "flagged": self.suspicion_score >= self.MAX_SCORE,
        })
        self._set_warning(event, repeat, penalty)
        self._check_burst(now, event)
        self._check_trust_cutoff()

    def _set_warning(self, event: str, repeat: int, penalty: int) -> None:
        if event not in WARNING_MESSAGES: return
        title, base_msg = WARNING_MESSAGES[event]
        msg = f"{base_msg} (occurrence ×{repeat})"
        warning = {
            "title": title, "message": msg, "event": event,
            "repeat": repeat, "penalty": penalty, "time": time.strftime("%H:%M:%S"),
        }
        self.active_warning = warning
        self.warning_history.append(warning)

    def _handle_tab_switch(self, now: float) -> None:
        if "tab_switched" in self.last_risk_time:
            if now - self.last_risk_time["tab_switched"] < 3: return
        self.last_risk_time["tab_switched"] = now
        self.tab_switch_count += 1

        if self.tab_switch_count == 1:
            self.suspicion_score = min(self.MAX_SCORE, self.suspicion_score + 50)
            self.active_warning = {
                "title": "🖥️ Tab Switch Detected",
                "message": "One more tab switch will TERMINATE your test.",
                "event": "tab_switched", "repeat": 1, "penalty": 50, "time": time.strftime("%H:%M:%S"),
            }
            self.warning_history.append(self.active_warning)
        elif self.tab_switch_count >= 2:
            self.test_terminated = True
            self.suspicion_score = self.MAX_SCORE

    def _check_burst(self, now: float, event: str) -> None:
        self.recent_events.append((now, event))
        self.recent_events = [(t, e) for t, e in self.recent_events if now - t <= self.burst_window]
        distinct = len(set(e for _, e in self.recent_events))
        if distinct >= self.burst_threshold and not self.burst_bonus_applied:
            self.burst_bonus_applied = True
            self.suspicion_score = min(self.MAX_SCORE, self.suspicion_score + 5)

    def _check_trust_cutoff(self) -> None:
        trust = self.get_trust_score()
        cutoff = self.MAX_SCORE // 2
        if trust < cutoff:
            if not self.alert_active:
                self.alert_active = True
                self.cutoff_breach_count += 1
                self.alert_messages.append({"time": time.strftime("%H:%M:%S"), "trust": trust, "message": "CRITICAL"})
        else:
            self.alert_active = False

    def get_trust_score(self) -> int:
        return max(0, 50 - self.suspicion_score)

    def is_flagged(self) -> bool:
        return self.suspicion_score >= self.MAX_SCORE or self.test_terminated

    def get_risk_level(self) -> str:
        if self.suspicion_score == 0:  return "NO RISK"
        if self.suspicion_score > 40: return "HIGH RISK"
        if self.suspicion_score > 25: return "MEDIUM RISK"
        if self.suspicion_score > 12: return "LOW RISK"
        return "NO RISK"

    def get_pattern_summary(self) -> dict:
        return {e: {"count": c} for e, c in self.violation_counts.items()}