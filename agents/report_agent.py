import json
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


class ReportAgent:

    def __init__(self, risk_agent, violation_agent):
        self.risk_agent      = risk_agent
        self.violation_agent = violation_agent

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def generate_reports(self, duration: float, avg_attention: float) -> None:
        """
        Write analytics.json and report.pdf to the outputs/ directory.

        Args:
            duration:      exam duration in seconds
            avg_attention: mean attention score (0–100) across the session
        """
        os.makedirs("outputs", exist_ok=True)

        risk_level      = self.risk_agent.get_risk_level()
        suspicion_score = self.risk_agent.suspicion_score
        trust_score     = self.risk_agent.get_trust_score()
        violations      = self.violation_agent.violations

        # ── JSON ──────────────────────────────────────────────────
        def json_serial(obj):
            """JSON serializer for objects not serializable by default json code"""
            try:
                from bson import ObjectId
                if isinstance(obj, ObjectId):
                    return str(obj)
            except ImportError:
                pass
            raise TypeError(f"Type {type(obj)} not serializable")

        analytics = {
            "duration":        round(duration, 2),
            "avg_attention":   round(avg_attention, 2),
            "violations":      violations,
            "violation_count": len(violations),
            "suspicion_score": suspicion_score,
            "max_score":       self.risk_agent.MAX_SCORE,
            "trust_score":     trust_score,
            "risk":            risk_level,
            "flagged":         self.risk_agent.is_flagged(),
            "timeline":        self.risk_agent.timeline,
            "warning_history": self.risk_agent.warning_history,
            "pattern_summary": self.risk_agent.get_pattern_summary(),
        }

        with open("outputs/analytics.json", "w") as f:
            json.dump(analytics, f, indent=2, default=json_serial)

        print("[Report] analytics.json saved.")

        # ── PDF ───────────────────────────────────────────────────
        self._write_pdf(analytics)
        print("[Report] report.pdf saved.")

    # ─────────────────────────────────────────────────────────────
    # PDF builder
    # ─────────────────────────────────────────────────────────────

    def _write_pdf(self, data: dict) -> None:
        pdf  = canvas.Canvas("outputs/report.pdf", pagesize=A4)
        w, h = A4

        # ── Header ────────────────────────────────────────────────
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(2 * cm, h - 2 * cm, "AI Proctor — Exam Session Report")

        pdf.setFont("Helvetica", 10)
        pdf.setFillColorRGB(0.4, 0.4, 0.4)
        pdf.drawString(2 * cm, h - 2.6 * cm, "Generated automatically by Video Proctor")

        # ── Divider ───────────────────────────────────────────────
        pdf.setStrokeColorRGB(0.8, 0.8, 0.8)
        pdf.line(2 * cm, h - 3 * cm, w - 2 * cm, h - 3 * cm)

        # ── Summary block ─────────────────────────────────────────
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(2 * cm, h - 3.8 * cm, "Session Summary")

        max_score = data.get("max_score", 30)
        flagged   = data.get("flagged", False)
        rows = [
            ("Exam Duration",    f"{data['duration']:.0f} seconds"),
            ("Average Attention",f"{data['avg_attention']:.1f} / 100"),
            ("Suspicion Score", f"{data['suspicion_score']} / {max_score}"),
            ("Trust Score",     f"{data['trust_score']} / {max_score}"),
            ("Risk Level",      data["risk"]),
            ("Flagged",         "YES ⚠️" if flagged else "No"),
            ("Total Violations", str(data["violation_count"])),
        ]

        pdf.setFont("Helvetica", 11)
        y = h - 4.5 * cm
        for label, value in rows:
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(2 * cm, y, f"{label}:")
            pdf.setFont("Helvetica", 11)
            pdf.drawString(7 * cm, y, value)
            y -= 0.7 * cm

        # ── Violations table ──────────────────────────────────────
        y -= 0.5 * cm
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(2 * cm, y, "Violation Log")
        y -= 0.6 * cm

        if not data["violations"]:
            pdf.setFont("Helvetica", 11)
            pdf.drawString(2 * cm, y, "No violations recorded.")
        else:
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(2 * cm,  y, "Time")
            pdf.drawString(5 * cm,  y, "Type")
            pdf.drawString(10 * cm, y, "Detail")
            y -= 0.5 * cm

            pdf.setFont("Helvetica", 10)
            for v in data["violations"]:
                if y < 2 * cm:          # new page if needed
                    pdf.showPage()
                    y = h - 2 * cm
                    pdf.setFont("Helvetica", 10)

                time_str   = str(v.get("time",   "")).replace("_", ":")
                vtype_str  = str(v.get("type",   ""))
                detail_str = str(v.get("detail", ""))[:40]

                pdf.drawString(2 * cm,  y, time_str)
                pdf.drawString(5 * cm,  y, vtype_str)
                pdf.drawString(10 * cm, y, detail_str)
                y -= 0.5 * cm

        pdf.save()