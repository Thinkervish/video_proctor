"""
Shared runtime state between:
- main.py (AI agents + OpenCV)
- server.py (FastAPI streaming)
"""
import os

# Latest webcam frame for MJPEG streaming
# Shared runtime state

latest_frame = None
risk_agent = None
violation_agent = None
latest_frame = None
latest_frame_time = 0
proctoring_active = False
side_frame = None
side_frame_time = 0
Assessment_id = "Sairam"
Email_id = "Jrsairam@5686"

# Dynamic Video Scores (Capped at 50)
risk_score = 50
trust_score = 0
violation_score = 42

# Dynamic Code Scores (Capped at 20)
code_risk_score = 0
code_trust_score = 20
code_violation_score = 0

def save_state():
    """Persist current scores to this file (state.py)."""
    file_path = __file__
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.startswith("risk_score ="):
                new_lines.append(f"risk_score = {risk_score}\n")
            elif line.startswith("trust_score ="):
                new_lines.append(f"trust_score = {trust_score}\n")
            elif line.startswith("violation_score ="):
                new_lines.append(f"violation_score = {violation_score}\n")
            elif line.startswith("code_risk_score ="):
                new_lines.append(f"code_risk_score = {code_risk_score}\n")
            elif line.startswith("code_trust_score ="):
                new_lines.append(f"code_trust_score = {code_trust_score}\n")
            elif line.startswith("code_violation_score ="):
                new_lines.append(f"code_violation_score = {code_violation_score}\n")
            else:
                new_lines.append(line)
        
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
    except Exception:
        pass