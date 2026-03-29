"""
code_agents/code_risk_agent.py
─────────────────────────────────────────────────────────────────────────────
Tracks code-submission violations and maintains a running risk score for the
code-analysis pipeline.

Scoring:
  - All code violations : flat 5 pts each
  - Max score cap       : 20  (candidate flagged at 20)
  - Tab switch          : +50 pts (1st → final warning, 2nd → test terminated)

Violations tracked:
  - ai_generated   : AI-generated code detected by AIDetectionAgent
  - plagiarism     : Plagiarism detected by PlagiarismAgent
  - tab_switched   : Candidate left the code exam window
"""

import time
from collections import defaultdict
import state


class CodeRiskAgent:

    # Maximum suspicion score for code-analysis violations
    MAX_SCORE = 20

    def get_trust_score(self) -> int:
        return max(0, 20 - self.suspicion_score)

    def __init__(self):
        self.suspicion_score = 0
        self.timeline        = []

        # All code violations carry a flat 5-point penalty
        self.weights = {
            "ai_generated": 5,
            "plagiarism":   5,
        }

        self.violation_counts = defaultdict(int)
        self.last_risk_time   = {}
        self.risk_cooldown    = 0   # code submissions are discrete; no cooldown needed

        # ── Tab switch: 2-strike special ─────────────────────────
        self.tab_switch_count = 0
        self.test_terminated  = False

        # ── Warning tracking ──────────────────────────────────────
        self.active_warning  = None
        self.warning_history = []

    # ─────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────

    def update_risk(self, event: str) -> dict:
        """
        Record a violation event and return the updated risk summary.

        Args:
            event: one of 'ai_generated', 'plagiarism', 'tab_switched'

        Returns:
            dict with current risk state
        """
        now = time.time()

        if self.test_terminated:
            return self._summary()

        if event == "tab_switched":
            self._handle_tab_switch(now)
            return self._summary()

        if event not in self.weights:
            return self._summary()

        # ── Flat 5-point penalty ──────────────────────────────────
        self.violation_counts[event] += 1
        repeat  = self.violation_counts[event]
        penalty = self.weights[event]   # always 5

        old = self.suspicion_score
        self.suspicion_score = min(self.MAX_SCORE, self.suspicion_score + penalty)

        entry = {
            "event":   event,
            "score":   self.suspicion_score,
            "time":    time.strftime("%H:%M:%S"),
            "repeat":  repeat,
            "penalty": penalty,
            "flagged": self.suspicion_score >= self.MAX_SCORE,
        }
        self.timeline.append(entry)
        print(
            f"[CODE RISK] {event} ×{repeat} → "
            f"+{penalty} → "
            f"suspicion: {old} → {self.suspicion_score}"
        )

        self._set_warning(event, repeat)

        if self.suspicion_score >= self.MAX_SCORE:
            print("[CODE RISK] ⚠️  Candidate FLAGGED — max code risk score reached.")

        return self._summary()

    # ─────────────────────────────────────────────────────────────
    # Tab switch 2-strike
    # ─────────────────────────────────────────────────────────────

    def _handle_tab_switch(self, now: float) -> None:
        if "tab_switched" in self.last_risk_time:
            if now - self.last_risk_time["tab_switched"] < 3:
                return
        self.last_risk_time["tab_switched"] = now
        self.tab_switch_count += 1

        if self.tab_switch_count == 1:
            self.suspicion_score = min(self.MAX_SCORE, self.suspicion_score + 50)
            entry = {
                "event":   "tab_switched",
                "score":   self.suspicion_score,
                "time":    time.strftime("%H:%M:%S"),
                "repeat":  1,
                "penalty": 50,
                "flagged": True,
                "note":    "WARNING: Next tab switch will terminate the test.",
            }
            self.timeline.append(entry)
            self.active_warning = {
                "title":   "🖥️ Tab Switch Detected",
                "message": "You left the code exam window. One more tab switch will TERMINATE your test.",
                "event":   "tab_switched",
                "repeat":  1,
                "penalty": 50,
                "time":    time.strftime("%H:%M:%S"),
            }
            self.warning_history.append(self.active_warning)
            print("[CODE TAB] 1st switch +50. FINAL WARNING.")

        elif self.tab_switch_count >= 2:
            self.test_terminated = True
            self.suspicion_score = self.MAX_SCORE
            entry = {
                "event":   "tab_switched",
                "score":   self.MAX_SCORE,
                "time":    time.strftime("%H:%M:%S"),
                "repeat":  2,
                "penalty": 50,
                "flagged": True,
                "note":    "TEST TERMINATED: 2nd tab switch.",
            }
            self.timeline.append(entry)
            print("[CODE TAB] 2nd switch → TEST TERMINATED ❌")

    # ─────────────────────────────────────────────────────────────
    # Warning builder
    # ─────────────────────────────────────────────────────────────

    def _set_warning(self, event: str, repeat: int) -> None:
        messages = {
            "ai_generated": ("🤖 AI-Generated Code Detected",
                             "Your submission appears to be AI-generated. This has been flagged."),
            "plagiarism":   ("📋 Plagiarism Detected",
                             "Your submission shows high similarity to existing code. This has been flagged."),
        }
        if event not in messages:
            return
        title, base_msg = messages[event]

        if repeat == 1:
            msg = base_msg
        elif repeat == 2:
            msg = f"{base_msg} (2nd occurrence — risk increasing)"
        else:
            msg = f"{base_msg} (Repeated violation ×{repeat} — HIGH risk)"

        warning = {
            "title":   title,
            "message": msg,
            "event":   event,
            "repeat":  repeat,
            "penalty": self.weights[event],
            "time":    time.strftime("%H:%M:%S"),
        }
        self.active_warning = warning
        self.warning_history.append(warning)
        print(f"[CODE WARNING] {title} — repeat ×{repeat}")

    # ─────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────

    def get_risk_level(self) -> str:
        if self.suspicion_score == 0:  return "NO RISK"
        if self.suspicion_score >= 20: return "FLAGGED"
        if self.suspicion_score >= 15: return "HIGH RISK"
        if self.suspicion_score >= 10: return "MEDIUM RISK"
        if self.suspicion_score >= 5:  return "LOW RISK"
        return "NO RISK"

    def is_flagged(self) -> bool:
        return self.suspicion_score >= self.MAX_SCORE or self.test_terminated

    def get_pattern_summary(self) -> dict:
        return {
            e: {"count": c}
            for e, c in self.violation_counts.items()
        }

    def _summary(self) -> dict:
        return {
            "suspicion_score":         self.suspicion_score,
            "code_agents_risk_score": self.suspicion_score,
            "max_score":       self.MAX_SCORE,
            "risk_level":      self.get_risk_level(),
            "flagged":         self.is_flagged(),
            "terminated":      self.test_terminated,
            "tab_switches":    self.tab_switch_count,
            "violations":      dict(self.violation_counts),
        }
