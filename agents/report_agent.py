import json
from reportlab.pdfgen import canvas

class ReportAgent:
    def __init__(self, risk_agent, violation_agent):
        self.risk_agent = risk_agent
        self.violation_agent = violation_agent

    def generate_reports(self, duration, avg_attention):
        risk_level = self.risk_agent.get_risk_level()
        suspicion_score = self.risk_agent.suspicion_score

        analytics = {
            "duration": duration,
            "avg_attention": avg_attention,
            "violations": self.violation_agent.violations,
            "suspicion_score": suspicion_score,
            "risk": risk_level,
            "timeline": self.risk_agent.timeline
        }

        with open("outputs/analytics.json", "w") as f:
            json.dump(analytics, f, indent=2)

        pdf = canvas.Canvas("outputs/report.pdf")
        pdf.drawString(50, 800, "AI Proctor Report")
        pdf.drawString(50, 760, f"Violations:{len(self.violation_agent.violations)}")
        pdf.drawString(50, 740, f"Suspicion Score:{suspicion_score}")
        pdf.drawString(50, 720, f"Risk:{risk_level}")
        pdf.save()