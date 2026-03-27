"""
main.py  (clean — single front cam only)
────────────────────────────────────────
Proctoring loop for front camera only.
"""

import cv2
import time
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



# ─────────────────────────────────────────────────────────────
# Initialize agents
# ─────────────────────────────────────────────────────────────
vision_agent    = VisionAgent()
attention_agent = AttentionAgent()

state.risk_agent      = RiskAgent()
state.violation_agent = ViolationAgent()

supervisor_agent = SupervisorAgent(state.risk_agent, state.violation_agent)
report_agent     = ReportAgent(state.risk_agent, state.violation_agent)

audio_agent    = AudioAgent()
spoofing_agent = SpoofingAgent()


# ─────────────────────────────────────────────────────────────
# Front-cam proctoring loop
# ─────────────────────────────────────────────────────────────
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

        frame = cv2.resize(frame, (640, 480))

        try:
            # ── Run AI agents ─────────────────────────────
            vision_data    = vision_agent.analyze_vision(frame)
            attention_data = attention_agent.analyze_attention(frame)
            audio_data     = audio_agent.analyze_audio()
            spoof_data     = spoofing_agent.analyze_spoofing(frame)

            # print(
            #     f"[FRONT] face={vision_data['face_visible']} | "
            #     f"multi={vision_data['multiple_people']} | "
            #     f"illegal={vision_data['illegal_objects']}"
            # )
            # 
            # print(
            #     f"[FRONT] attention={attention_data['attention']} | "
            #     f"drowsy={attention_data['drowsy']} | "
            #     f"head={attention_data['head_turn']} | "
            #     f"mouth={attention_data['mouth_open']}"
            # )
            # 
            # print(
            #     f"[AUDIO] talking={audio_data['talking']} | "
            #     f"noise={audio_data['loud_noise']}"
            # )

            # ── Supervisor decision engine ───────────────
            supervisor_agent.supervise(
                vision_data,
                attention_data,
                audio_data,
                spoof_data,
                frame,
            )

            cv2.putText(
                frame,
                f"Suspicion:{state.risk_agent.suspicion_score}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # ── Periodic Report Update (every 30 frames) ───────
            state.frame_count = getattr(state, "frame_count", 0) + 1
            if state.frame_count % 30 == 0:
                avg_attention = int(np.mean(attention_scores)) if attention_scores else 0
                report_agent.generate_reports(elapsed, avg_attention)
                print(f"[FRONT] Periodic report updated (Frame {state.frame_count})")

            state.latest_frame = frame.copy()
            pass

        except Exception as e:
            print("Proctoring error:", e)
            continue

        time.sleep(0.03)

    # ── Clean shutdown ───────────────────────────────
    audio_agent.stop()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    avg_attention = int(np.mean(attention_scores)) if attention_scores else 0
    report_agent.generate_reports(elapsed, avg_attention)

    # ── Final Score Sync ─────────────────────────────
    state.risk_score      = state.risk_agent.suspicion_score
    state.trust_score     = state.risk_agent.get_trust_score()
    state.violation_score = sum(state.risk_agent.violation_counts.values())
    state.save_state()

    print("FRONT CAM PROCTORING DONE ✅")


# ─────────────────────────────────────────────────────────────
# Code Analysis (CLI testing)
# ─────────────────────────────────────────────────────────────
