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
latest_frame = None
latest_frame_time = 0
proctoring_active =False
side_frame = None
side_frame_time = 0
Assessment_id = "Sairam"
Email_id = "Jrsairam@5686"