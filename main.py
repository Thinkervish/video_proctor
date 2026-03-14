import cv2
import time
import numpy as np
import threading
import uvicorn
import state
from agents.vision_agent import VisionAgent
from agents.attention_agent import AttentionAgent
from agents.violation_agent import ViolationAgent
from agents.supervisor_agent import SupervisorAgent
from agents.report_agent import ReportAgent
from agents.risk_agent import RiskAgent
from agents.audio_agent import AudioAgent
from agents.spoofing_agent import SpoofingAgent
from server import app
from api.code_routes import router as code_router



TEST_DURATION = 60
app.include_router(code_router)

# Initialize all agents
vision_agent = VisionAgent()
attention_agent = AttentionAgent()
state.risk_agent = RiskAgent()
state.violation_agent = ViolationAgent()
supervisor_agent = SupervisorAgent(state.risk_agent, state.violation_agent)
report_agent = ReportAgent(state.risk_agent, state.violation_agent)
audio_agent = AudioAgent()
spoofing_agent  = SpoofingAgent()

latest_frame = None  # Global for FastAPI

def run_proctoring():
    global latest_frame
    cap = cv2.VideoCapture(0)
    attention_scores = []
    
    # Start Audio Listener asynchronously
    audio_agent.start()
    
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        vision_data = vision_agent.analyze_vision(frame)
        attention_data = attention_agent.analyze_attention(frame)
        audio_data = audio_agent.analyze_audio()
        spoof_data    = spoofing_agent.analyze_spoofing(frame)


        supervisor_agent.supervise(vision_data, attention_data, audio_data, frame, spoof_data = spoof_data)

        attention_scores.append(attention_data["attention"])

        elapsed = int(time.time() - start)
        cv2.putText(frame, f"Suspicion:{state.risk_agent.suspicion_score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Update global frame for the web server to hook into
        state.latest_frame = frame.copy()
        
        # Optional local debugging window
        cv2.imshow("Agentic Proctor", frame)

        if elapsed > TEST_DURATION:
            break

    # Clean shut down
    audio_agent.stop()
    cap.release()
    cv2.destroyAllWindows()

    avg_attention = int(np.mean(attention_scores)) if attention_scores else 0
    report_agent.generate_reports(elapsed, avg_attention)
    print("PROCTORING SESSION DONE ✅ Reports Generated.")


if __name__ == "__main__":
    # 1. Start AI Agent Proctoring in a background thread
    proctor_thread = threading.Thread(target=run_proctoring, daemon=True)
    proctor_thread.start()

    # 2. Start FastAPI Server on main thread
    print("Starting Web Dashboard on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)