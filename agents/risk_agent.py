"""
agents/risk_agent.py  (updated — side cam aware)
─────────────────────────────────────────────────────────────────────────────
Added:
  • "looking_down"  weight + warning message
    (candidate looking at lap / phone / notes — detected by side cam pitch)

Everything else is unchanged from your original.
"""

import time
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
TRUST_CUTOFF = 50   # Trust below this → proctor alert
# ─────────────────────────────────────────────────────────────────────────────

WARNING_MESSAGES = {
    "illegal_object":    ("📱 Illegal Object Detected",     "Please remove all prohibited items from view immediately."),
    "multiple_people":   ("👥 Multiple People Detected",    "Only the candidate should be visible on camera."),
    "face_not_visible":  ("👤 Face Not Visible",            "Please ensure your face is clearly visible on camera."),
    "head_turned":       ("↩️ Head Turn Detected",          "Please keep your eyes on the screen at all times."),
    "talking":           ("🗣️ Talking Detected",            "Please maintain silence during the exam."),
    "loud_noise":        ("🔊 Loud Noise Detected",         "Suspicious audio activity has been flagged."),
    "drowsy":            ("😴 Drowsiness Detected",         "Please stay alert and focused on your exam."),
    "mouth_open":        ("👄 Whispering Suspected",        "Unusual mouth movement detected. Please stay silent."),
    "low_attention":     ("😶 Low Attention",               "Please focus on the exam screen."),
    "spoofing_attempt":  ("🎭 Spoofing Attempt Detected",   "Identity verification failed. This has been flagged."),
    # ── NEW: side-cam-specific ───────────────────────────────────────────
    "looking_down":      ("👀 Looking Down Detected",       "Please keep your eyes on the screen and avoid looking down."),
}


class RiskAgent:
    def __init__(self):
        self.suspicion_score = 0
        self.timeline        = []

        self.weights = {
            "illegal_object":   25,
            "multiple_people":  30,
            "low_attention":    15,
            "drowsy":           10,
            "face_not_visible": 20,
            "head_turned":      15,
            "talking":          25,
            "loud_noise":       15,
            "spoofing_attempt": 50,
            "mouth_open":       20,
            # ── NEW ──────────────────────────────────────────────────
            "looking_down":     20,   # same weight as face_not_visible
            # tab_switched handled separately
        }

        # repeat 1→×1.0  repeat 2→×1.5  repeat 3→×2.0  repeat 4+→×2.5
        self.escalation_multipliers = [1.0, 1.5, 2.0, 2.5]
        self.violation_counts       = defaultdict(int)
        self.last_risk_time         = {}
        self.risk_cooldown          = 5      # seconds between same-event logs

        # ── TAB SWITCH: 2-strike special ────────────────────────────
        self.tab_switch_count  = 0
        self.test_terminated   = False

        # ── Active warning ───────────────────────────────────────────
        self.active_warning  = None
        self.warning_history = []

        # ── Burst detection ──────────────────────────────────────────
        self.burst_window        = 60
        self.burst_threshold     = 4
        self.recent_events       = []
        self.burst_bonus_applied = False

        # ── Trust cutoff alert ───────────────────────────────────────
        self.alert_active        = False
        self.alert_triggered_at  = None
        self.alert_messages      = []
        self.cutoff_breach_count = 0

    # ── MAIN ENTRY ───────────────────────────────────────────────────────────
    def update_risk(self, event):
        now = time.time()
        if self.test_terminated:
            return

        if event == "tab_switched":
            self._handle_tab_switch(now)
            return

        # Cooldown
        if event in self.last_risk_time:
            if now - self.last_risk_time[event] < self.risk_cooldown:
                return
        self.last_risk_time[event] = now

        if event not in self.weights:
            return

        # ── Escalating penalty ───────────────────────────────────────
        self.violation_counts[event] += 1
        repeat    = self.violation_counts[event]
        mult_idx  = min(repeat - 1, len(self.escalation_multipliers) - 1)
        multiplier = self.escalation_multipliers[mult_idx]
        penalty   = int(self.weights[event] * multiplier)

        old = self.suspicion_score
        self.suspicion_score = min(100, self.suspicion_score + penalty)

        self.timeline.append({
            "event":      event,
            "score":      self.suspicion_score,
            "time":       time.strftime("%H:%M:%S"),
            "repeat":     repeat,
            "penalty":    penalty,
            "multiplier": multiplier,
        })
        print(
            f"[RISK] {event} ×{repeat} → "
            f"+{penalty} (×{multiplier}) → "
            f"suspicion: {old} → {self.suspicion_score}"
        )

        self._set_warning(event, repeat, penalty)
        self._check_burst(now, event)
        self._check_trust_cutoff()

    # ── WARNING BUILDER ──────────────────────────────────────────────────────
    def _set_warning(self, event, repeat, penalty):
        if event not in WARNING_MESSAGES:
            return
        title, base_msg = WARNING_MESSAGES[event]

        if repeat == 1:
            msg = base_msg
        elif repeat == 2:
            msg = f"{base_msg} (2nd occurrence — suspicion increasing)"
        elif repeat == 3:
            msg = f"{base_msg} (3rd occurrence — HIGH suspicion)"
        else:
            msg = f"{base_msg} (Repeated violation ×{repeat} — CRITICAL suspicion)"

        warning = {
            "title":   title,
            "message": msg,
            "event":   event,
            "repeat":  repeat,
            "penalty": penalty,
            "time":    time.strftime("%H:%M:%S"),
        }
        self.active_warning = warning
        self.warning_history.append(warning)
        print(f"[WARNING] {title} — repeat ×{repeat} penalty +{penalty}")

    # ── TAB SWITCH 2-STRIKE ──────────────────────────────────────────────────
    def _handle_tab_switch(self, now):
        if "tab_switched" in self.last_risk_time:
            if now - self.last_risk_time["tab_switched"] < 3:
                return
        self.last_risk_time["tab_switched"] = now
        self.tab_switch_count += 1

        if self.tab_switch_count == 1:
            self.suspicion_score = min(100, self.suspicion_score + 50)
            self.timeline.append({
                "event": "tab_switched", "score": self.suspicion_score,
                "time": time.strftime("%H:%M:%S"), "repeat": 1, "penalty": 50,
                "multiplier": 1.0,
                "note": "WARNING: Next tab switch will terminate the test.",
            })
            self.active_warning = {
                "title":   "🖥️ Tab Switch Detected",
                "message": "You left the exam window. One more tab switch will TERMINATE your test.",
                "event":   "tab_switched", "repeat": 1, "penalty": 50,
                "time":    time.strftime("%H:%M:%S"),
            }
            self.warning_history.append(self.active_warning)
            print("[TAB] 1st switch +50. FINAL WARNING.")
            self._check_trust_cutoff()

        elif self.tab_switch_count >= 2:
            self.test_terminated = True
            self.suspicion_score = 100
            self.timeline.append({
                "event": "tab_switched", "score": 100,
                "time": time.strftime("%H:%M:%S"), "repeat": 2, "penalty": 100,
                "multiplier": 2.0, "note": "TEST TERMINATED: 2nd tab switch.",
            })
            print("[TAB] 2nd switch → TEST TERMINATED ❌")
            self._check_trust_cutoff()

    # ── BURST DETECTION ──────────────────────────────────────────────────────
    def _check_burst(self, now, event):
        self.recent_events.append((now, event))
        self.recent_events = [
            (t, e) for t, e in self.recent_events if now - t <= self.burst_window
        ]
        distinct = len(set(e for _, e in self.recent_events))
        if distinct >= self.burst_threshold and not self.burst_bonus_applied:
            self.burst_bonus_applied = True
            self.suspicion_score = min(100, self.suspicion_score + 20)
            self.timeline.append({
                "event": "BURST_PATTERN", "score": self.suspicion_score,
                "time": time.strftime("%H:%M:%S"),
                "repeat": 1, "penalty": 20, "multiplier": 1.0,
            })
            print(f"[RISK] ⚡ BURST — {distinct} violations in {self.burst_window}s → +20")

    # ── TRUST CUTOFF ─────────────────────────────────────────────────────────
    def _check_trust_cutoff(self):
        trust = self.get_trust_score()
        if trust < TRUST_CUTOFF:
            if not self.alert_active:
                self.alert_active        = True
                self.alert_triggered_at  = time.strftime("%H:%M:%S")
                self.cutoff_breach_count += 1
                self.alert_messages.append({
                    "time":      self.alert_triggered_at,
                    "trust":     trust,
                    "suspicion": self.suspicion_score,
                    "message":   f"TRUST CRITICAL: {trust}%",
                    "breach":    self.cutoff_breach_count,
                })
                print(f"[ALERT] Trust {trust}% — breach #{self.cutoff_breach_count}")
        else:
            if self.alert_active:
                self.alert_active = False
                print(f"[ALERT CLEARED] Trust recovered → {trust}%")

    # ── HELPERS ──────────────────────────────────────────────────────────────
    def get_trust_score(self):
        return max(0, 100 - self.suspicion_score)

    def get_latest_alert(self):
        return self.alert_messages[-1] if self.alert_messages else None

    def get_pattern_summary(self):
        return {
            e: {
                "count":      c,
                "multiplier": self.escalation_multipliers[min(c - 1, 3)],
            }
            for e, c in self.violation_counts.items()
        }

    def get_risk_level(self):
        if self.suspicion_score > 80: return "HIGH RISK"
        if self.suspicion_score > 50: return "MEDIUM RISK"
        if self.suspicion_score > 25: return "LOW RISK"
        return "NORMAL"