"""
Shared runtime state between:
- main.py (AI agents + OpenCV)
- server.py (FastAPI streaming)
"""

# Latest webcam frame for MJPEG streaming
# Shared runtime state

latest_frame = None
risk_agent = None
violation_agent = None

proctoring_active =False

Assessment_id = "Sairam"
Email_id = "Jrsairam@5686"