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
Assessment_id = ""
Email_id = ""

# Dynamic Video Scores (Capped at 50)
risk_score = 25
trust_score = 25
violation_score = 6

# Dynamic Code Scores (Capped at 20)
code_risk_score = 0
code_trust_score = 20
code_violation_score = 0

def save_state():
    """Persist current scores to state.json."""
    import json
    data = {
        "video": {
            "risk_score": risk_score,
            "trust_score": trust_score,
            "violation_score": violation_score
        },
        "code": {
            "risk_score": code_risk_score,
            "trust_score": code_trust_score,
            "violation_score": code_violation_score
        },
        "assessment_id": Assessment_id,
        "email_id": Email_id
    }
    try:
        with open("state.json", "w") as f:
            json.dump(data, f, indent=4)
        # print("[State] Persistent scores saved to state.json.")
    except Exception as e:
        print(f"[State] Error saving state: {e}")

def load_state():
    """Load scores from state.json if it exists."""
    import json
    global risk_score, trust_score, violation_score
    global code_risk_score, code_trust_score, code_violation_score
    global Assessment_id, Email_id
    
    if os.path.exists("state.json"):
        try:
            with open("state.json", "r") as f:
                data = json.load(f)
            
            v = data.get("video", {})
            risk_score      = v.get("risk_score", risk_score)
            trust_score     = v.get("trust_score", trust_score)
            violation_score = v.get("violation_score", violation_score)
            
            c = data.get("code", {})
            code_risk_score      = c.get("risk_score", code_risk_score)
            code_trust_score     = c.get("trust_score", code_trust_score)
            code_violation_score = c.get("violation_score", code_violation_score)
            
            Assessment_id = data.get("assessment_id", Assessment_id)
            Email_id      = data.get("email_id", Email_id)
            # print("[State] Scores loaded from state.json.")
        except Exception as e:
            print(f"[State] Error loading state: {e}")

# Load state on module import
load_state()