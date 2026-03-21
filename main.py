"""
main.py  (updated — dual cam: front cam + Agora side cam)
─────────────────────────────────────────────────────────────────────────────
Two proctoring loops run concurrently in separate threads:

  run_proctoring()        — front cam  (state.latest_frame)
  run_side_proctoring()   — side cam   (state.side_frame via Agora)

Both share the same RiskAgent, ViolationAgent, SupervisorAgent.
Side cam uses SideAttentionAgent (tuned for 90° profile view).
Audio is only processed in the front-cam loop to avoid double-counting.
"""

import cv2
import time
import threading
import numpy as np

import state
from agents.vision_agent     import VisionAgent
from agents.attention_agent  import AttentionAgent   # handles both cams
from agents.violation_agent  import ViolationAgent
from agents.supervisor_agent import SupervisorAgent
from agents.report_agent     import ReportAgent
from agents.risk_agent       import RiskAgent
from agents.audio_agent      import AudioAgent
from agents.spoofing_agent   import SpoofingAgent


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

# Separate VisionAgent instance for side cam
# (YOLO is not thread-safe with a single instance)
side_vision_agent   = VisionAgent()
side_spoofing_agent = SpoofingAgent()


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
        camera_type = getattr(state, "camera_type", "laptop")

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


# ─────────────────────────────────────────────────────────────────────────────
# Side-cam proctoring loop  (reads from state.side_frame populated by Agora)
# ─────────────────────────────────────────────────────────────────────────────
def run_side_proctoring():
    """
    Runs in its own thread.
    Reads frames delivered by agora_receiver.py into state.side_frame.
    Uses SideAttentionAgent (pitch-aware, profile-tuned) instead of
    the standard AttentionAgent.
    Audio checks are skipped here — handled by run_proctoring() only.
    """
    print("[SIDE CAM] Proctoring loop started — waiting for Agora frames...")

    # Wait until the first side-cam frame arrives (up to 30 s)
    wait_start = time.time()
    while getattr(state, "side_frame", None) is None:
        if time.time() - wait_start > 30:
            print("[SIDE CAM] ⚠️  No Agora frame received in 30 s. Loop exiting.")
            return
        time.sleep(0.2)

    print("[SIDE CAM] ✅ First frame received — analysis running.")

    while state.proctoring_active:
        side_frame     = getattr(state, "side_frame", None)
        side_frame_age = time.time() - getattr(state, "side_frame_time", 0)

        if side_frame is None or side_frame_age > 3.0:
            # Stale / no frame — log as face not visible after 3 s gap
            if side_frame_age > 3.0 and side_frame is not None:
                print("[SIDE CAM] ⚠️  Frame stale — side cam may be disconnected.")
            time.sleep(0.05)
            continue

        frame = cv2.resize(side_frame, (640, 480))

        # ── Run AI agents (side cam) ─────────────────────────────
        vision_data    = side_vision_agent.analyze_vision(frame, camera_type="mobile")
        attention_data = attention_agent.analyze(frame, camera_type="mobile")  # "mobile" branch
        spoof_data     = side_spoofing_agent.analyze_spoofing(frame, camera_type="mobile")

        # No audio — side cam shares the same audio environment;
        # audio is already processed in run_proctoring()
        audio_data = {"talking": False, "loud_noise": False}

        print(
            f"[SIDE] face={vision_data['face_visible']} | "
            f"multi={vision_data['multiple_people']} | "
            f"illegal={vision_data['illegal_objects']}"
        )
        print(
            f"[SIDE] attention={attention_data['attention']} | "
            f"drowsy={attention_data['drowsy']} | "
            f"head_turn={attention_data['head_turn']} | "
            f"looking_down={attention_data.get('looking_down', False)} | "
            f"mouth={attention_data['mouth_open']}"
        )

        # ── Supervisor decision engine ───────────────────────────
        supervisor_agent.supervise(
            vision_data, attention_data, audio_data,
            frame, camera_type="mobile",              # ← marks this as side cam
            spoof_data=spoof_data,
        )

        # Optional: show side cam debug window
        cv2.putText(
            frame,
            f"[SIDE] Suspicion:{state.risk_agent.suspicion_score}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2,
        )
        state.side_frame_annotated = frame.copy()

        time.sleep(0.03)   # ~30 fps

    print("[SIDE CAM] Proctoring loop stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: start both loops together
# ─────────────────────────────────────────────────────────────────────────────
_side_thread = None

def start_side_proctoring():
    global _side_thread
    if _side_thread and _side_thread.is_alive():
        return
    _side_thread = threading.Thread(target=run_side_proctoring, daemon=True)
    _side_thread.start()