"""
server.py
─────────────────────────────────────────────────────────────────────────────
FastAPI server combining video proctoring and code analysis.
Side camera removed. One endpoint for code analysis via supervisor.
"""

import base64
import json
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import time
import threading

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import state
from main import run_proctoring
from code_agents.code_supervisor_agent import CodeSupervisorAgent
from code_agents.plagiarism_agent import PlagiarismAgent
from code_agents.ai_detection_agent import AIDetectionAgent
from Connections.ViolationLogsDB import CodeEvaluation_collection


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Video Proctor",
    description="AI-powered exam proctoring + code analysis API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs/evidence", exist_ok=True)
app.mount("/evidence", StaticFiles(directory="outputs/evidence"), name="evidence")
templates = Jinja2Templates(directory="templates")

plagiarism_agent = PlagiarismAgent()
ai_agent = AIDetectionAgent()

_supervisor = CodeSupervisorAgent(plagiarism_agent, ai_agent)
# Supervisor instance for code analysis


@app.on_event("shutdown")
def shutdown_event():
    print("[Server] Shutdown signal received. Cleaning up...")
    if getattr(state, "proctoring_active", False):
        state.proctoring_active = False
        # Wait a moment for the thread to finish and save state
        if proctor_thread and proctor_thread.is_alive():
            proctor_thread.join(timeout=2.0)
    print("[Server] Shutdown complete.")



# ── Ensure state fields exist ────────────────────────────────────────────────
if not hasattr(state, "latest_frame"):       state.latest_frame       = None
if not hasattr(state, "latest_frame_time"):  state.latest_frame_time  = 0.0
if not hasattr(state, "proctoring_active"):  state.proctoring_active  = False
if not hasattr(state, "Assessment_id"):      state.Assessment_id      = None
if not hasattr(state, "Email_id"):           state.Email_id           = None
state.risk_score         = 0
state.trust_score        = 50
state.violation_score    = 0
state.code_risk_score      = 0
state.code_trust_score     = 20
state.code_violation_score = 0


# ---------------------------------------------------------------------------
# Proctoring thread management
# ---------------------------------------------------------------------------

proctor_thread: threading.Thread | None = None


def start_proctoring() -> None:
    global proctor_thread
    if proctor_thread is None or not proctor_thread.is_alive():
        print("[Server] Starting AI Proctoring Thread...")
        proctor_thread = threading.Thread(target=run_proctoring, daemon=True)
        proctor_thread.start()
    else:
        pass


# ---------------------------------------------------------------------------
# MJPEG stream helper
# ---------------------------------------------------------------------------

def _generate_frames():
    """MJPEG generator — reads front-cam frames from state."""
    while True:
        frame = state.latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


# ---------------------------------------------------------------------------
# VIDEO PROCTOR ROUTES
# ---------------------------------------------------------------------------

@app.get("/", summary="Live dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/video", summary="MJPEG front-cam stream")
def video_feed():
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/analytics", summary="Live analytics snapshot")
def get_analytics():
    suspicion_score = 0
    risk_level      = "NORMAL"
    trust_score     = 30
    flagged         = False
    timeline        = []
    violations      = []

    if state.risk_agent is not None:
        suspicion_score = state.risk_agent.suspicion_score
        risk_level      = state.risk_agent.get_risk_level()   # fixed: was .risk_level (no such attribute)
        trust_score     = state.risk_agent.get_trust_score()
        flagged         = state.risk_agent.is_flagged()
        timeline        = state.risk_agent.timeline

    if state.violation_agent is not None:
        violations = state.violation_agent.violations

    return JSONResponse({
        "suspicion_score":   suspicion_score,
        "max_score":         30,
        "trust_score":       trust_score,
        "risk_level":        risk_level,
        "flagged":           flagged,
        "violation_count":   len(violations),
        "violations":        violations,
        "timeline":          timeline,
        "proctoring_active": getattr(state, "proctoring_active", False),
        "assessment_id":     getattr(state, "Assessment_id", None),
        "email_id":          getattr(state, "Email_id", None),
    })


@app.post("/video/frame", summary="Receive a base64-encoded frame from browser")
async def receive_frame(request: Request):
    state.proctoring_active = True

    data = await request.json()

    image_b64           = data.get("image", "")
    state.Assessment_id = data.get("assessment_id", "")
    state.Email_id      = data.get("email_id", "")

    try:
        header, encoded = image_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(
                {"status": "error", "detail": "Failed to decode image"},
                status_code=400,
            )

        state.latest_frame      = frame
        state.latest_frame_time = time.time()

    except Exception as exc:
        return JSONResponse(
            {"status": "error", "detail": str(exc)},
            status_code=400,
        )

    start_proctoring()

    return {
        "status":        "frame received",
        "assessment_id": state.Assessment_id,
        "email_id":      state.Email_id,
    }


@app.post("/stop", summary="Stop the current proctoring session")
def stop_proctoring():
    if not getattr(state, "proctoring_active", False):
        return {"status": "proctoring already stopped"}
    
    print("[VideoProctor] STOP signal received. Finalizing session...")
    state.proctoring_active = False
    return {"status": "proctoring stopped"}



@app.get("/report", summary="Download the latest exam report (JSON)")
def get_report():
    report_path = "outputs/analytics.json"
    if not os.path.exists(report_path):
        return JSONResponse(
            {"status": "no_report", "detail": "No report generated yet."},
            status_code=404,
        )
    with open(report_path, "r") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/health", summary="Health check")
def health_check():
    return {
        "status":            "ok",
        "proctoring_active": getattr(state, "proctoring_active", False),
        "has_frame":         state.latest_frame is not None,
    }



@app.post("/Code/Checker", summary="Analyse candidate code for anomalies")
async def code_checker(request: Request):
    
    data = await request.json() 

    code         = data.get("code")
    email = data.get("email")
    language = data.get("language")
    question_id   = data.get("question_id")
    assessment_id = data.get("assessment_id")
    result = _supervisor.analyze(code , language)
    print(result)

    val  = {
        "code" : code , 
        "language" :language,
        "email": email,
        "question_id" : question_id , 
        "assessment_id" : assessment_id , 
        "result" : result 

    }
    CodeEvaluation_collection.insert_one(val)
    

@app.post("/webcam/score/store")
async def store_scores(request: Request):
    data = await request.json()
    data = {
        "assessment_id" : data.get("assessment_id") ,
        "email" : data.get("email") ,
        "risk_score" : state.risk_score,
        "trust_score" : state.trust_score,
        "violation_score" : state.violation_score,
        "code_risk_score" : state.code_risk_score,
        "code_trust_score" : state.code_trust_score,
        "code_violation_score" : state.code_violation_score
    }
    
    Risk_Score_DB.insert_one(data)
    return {"status": True}