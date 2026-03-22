"""
main.py  (updated — front cam)
─────────────────────────────────────────────────────────────────────────────
proctoring loops run concurrently in separate threads:

  run_proctoring()        — front cam  (state.latest_frame)
  


.
Audio is only processed in the front-cam loop to avoid double-counting.
"""

import cv2
import time
import threading
import numpy as np

import state
from agents.vision_agent     import VisionAgent
from agents.attention_agent  import AttentionAgent   
from agents.violation_agent  import ViolationAgent
from agents.supervisor_agent import SupervisorAgent
from agents.report_agent     import ReportAgent
from agents.risk_agent       import RiskAgent
from agents.audio_agent      import AudioAgent
from agents.spoofing_agent   import SpoofingAgent
from code_agents.plagiarism_agent import PlagiarismAgent
from code_agents.ai_detection_agent import AIDetectionAgent
from code_agents.code_supervisor_agent import CodeSupervisorAgent


# ─────────────────────────────────────────────────────────────────────────────
# Shared agents  (initialised once, used by both loops)
# ─────────────────────────────────────────────────────────────────────────────
vision_agent    = VisionAgent()
attention_agent = AttentionAgent()   # single instance — routes internally

state.risk_agent      = RiskAgent()
state.violation_agent = ViolationAgent()
supervisor_agent      = SupervisorAgent(state.risk_agent, state.violation_agent)
report_agent          = ReportAgent(state.risk_agent, state.violation_agent)
audio_agent           = AudioAgent()
spoofing_agent        = SpoofingAgent()




# ─────────────────────────────────────────────────────────────────────────────
# Front-cam proctoring loop
# ─────────────────────────────────────────────────────────────────────────────
def run_proctoring():
    attention_scores = []
    audio_agent.start()
    start   = time.time()
    elapsed = 0

    while state.proctoring_active:
        frame     = state.latest_frame
        frame_age = time.time() - state.latest_frame_time

        if frame is None or frame_age > 2.0:
            time.sleep(0.03)
            continue

        frame       = cv2.resize(frame, (640, 480))
        camera_type = getattr(state, "laptop")

        # ── Run AI agents ────────────────────────────────────────
        vision_data    = vision_agent.analyze_vision(frame, camera_type)
        attention_data = attention_agent.analyze(frame, camera_type)
        audio_data     = audio_agent.analyze_audio()
        spoof_data     = spoofing_agent.analyze_spoofing(frame, camera_type)

        print(
            f"[FRONT] face={vision_data['face_visible']} | "
            f"multi={vision_data['multiple_people']} | "
            f"illegal={vision_data['illegal_objects']}"
        )
        print(
            f"[FRONT] attention={attention_data['attention']} | "
            f"drowsy={attention_data['drowsy']} | "
            f"head={attention_data['head_turn']} | "
            f"mouth={attention_data['mouth_open']}"
        )
        print(f"[AUDIO] talking={audio_data['talking']} | noise={audio_data['loud_noise']}")

        # ── Supervisor decision engine ───────────────────────────
        supervisor_agent.supervise(
            vision_data, attention_data, audio_data,
            frame, camera_type, spoof_data=spoof_data,
        )

        attention_scores.append(attention_data["attention"])
        elapsed = int(time.time() - start)

        cv2.putText(
            frame,
            f"Suspicion:{state.risk_agent.suspicion_score}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        state.latest_frame = frame.copy()
        cv2.imshow("Agentic Proctor — Front Cam", frame)
        time.sleep(0.03)

    # ── Clean shutdown ───────────────────────────────────────────
    audio_agent.stop()
    cv2.destroyAllWindows()
    avg_attention = int(np.mean(attention_scores)) if attention_scores else 0
    report_agent.generate_reports(elapsed, avg_attention)
    print("FRONT CAM PROCTORING DONE ✅")



if __name__ == "__main__":
    plagiarism_agent = PlagiarismAgent()
    ai_agent = AIDetectionAgent()

    supervisor = CodeSupervisorAgent(plagiarism_agent, ai_agent)

    while True:
        code = input("Enter code (or 'exit'): ")

        if code.lower() == "exit":
            break

        language = input("Enter language (python/java/cpp): ")

        supervisor.analyze(code, language)