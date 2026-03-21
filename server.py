import base64
import os
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
from api.code_routes import router as code_router


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

# Static files & templates
os.makedirs("outputs/evidence", exist_ok=True)
app.mount("/evidence", StaticFiles(directory="outputs/evidence"), name="evidence")
templates = Jinja2Templates(directory="templates")

# Wire in code-analysis routes  (/code/...)
app.include_router(code_router)


# ---------------------------------------------------------------------------
# Proctoring thread management
# ---------------------------------------------------------------------------

proctor_thread: threading.Thread | None = None


def start_proctoring() -> None:
    """Start the proctoring background thread (idempotent — safe to call repeatedly)."""
    global proctor_thread
    if proctor_thread is None or not proctor_thread.is_alive():
        print("[Server] Starting AI Proctoring Thread...")
        proctor_thread = threading.Thread(target=run_proctoring, daemon=True)
        proctor_thread.start()
    else:
        print("[Server] Proctoring thread already running.")


# ---------------------------------------------------------------------------
# MJPEG stream helper
# ---------------------------------------------------------------------------

def generate_frames():
    """Yield MJPEG frames from the shared state buffer."""
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
# ── PROCTORING ROUTES ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@app.get("/", response_class=None, summary="Live dashboard")
async def dashboard(request: Request):
    """Serve the live monitoring dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/video", summary="MJPEG webcam stream")
def video_feed():
    """Stream the live annotated webcam feed as MJPEG."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/analytics", summary="Live analytics snapshot")
def get_analytics():
    """
    Return a JSON snapshot of the current proctoring session state.

    Response:
        {
          "suspicion_score": int,
          "risk_level":      str,   // LOW / MEDIUM / HIGH
          "violation_count": int,
          "violations":      list,
          "timeline":        list,
          "proctoring_active": bool,
          "assessment_id":   str | null,
          "email_id":        str | null,
        }
    """
    suspicion_score = 0
    risk_level      = "LOW"
    timeline        = []
    violations      = []

    if state.risk_agent is not None:
        suspicion_score = state.risk_agent.suspicion_score
        risk_level      = state.risk_agent.risk_level
        timeline        = state.risk_agent.timeline

    if state.violation_agent is not None:
        violations = state.violation_agent.violations

    return JSONResponse({
        "suspicion_score":   suspicion_score,
        "risk_level":        risk_level,
        "violation_count":   len(violations),
        "violations":        violations,
        "timeline":          timeline,
        "proctoring_active": getattr(state, "proctoring_active", False),
        "assessment_id":     getattr(state, "Assessment_id", None),
        "email_id":          getattr(state, "Email_id", None),
    })


@app.post("/video/frame", summary="Receive a base64-encoded frame from browser")
async def receive_frame(request: Request):
    """
    Accept a frame sent from the browser (base64 JPEG), decode it,
    store it in shared state, and start the proctoring thread if needed.

    Request body (JSON):
        {
          "image":         str,   // base64 data URL  e.g. "data:image/jpeg;base64,..."
          "assessment_id": str,
          "email_id":      str,
        }
    """
    state.proctoring_active = True

    data = await request.json()

    image_b64      = data.get("image", "")
    state.Assessment_id = data.get("assessment_id", "")
    state.Email_id      = data.get("email_id", "")

    # Decode base64 → OpenCV frame
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
    """
    Signal the proctoring loop to stop.
    The report_agent will generate the final PDF + JSON on shutdown.
    """
    state.proctoring_active = False
    return {"status": "proctoring stopped"}


@app.get("/report", summary="Download the latest exam report (JSON)")
def get_report():
    """
    Return the latest analytics.json report if it exists.
    Useful for the admin dashboard after an exam ends.
    """
    report_path = "outputs/analytics.json"
    if not os.path.exists(report_path):
        return JSONResponse(
            {"status": "no_report", "detail": "No report generated yet."},
            status_code=404,
        )
    import json
    with open(report_path, "r") as f:
        data = json.load(f)
    return JSONResponse(data)


@app.get("/health", summary="Health check")
def health_check():
    """Quick liveness check — returns 200 if server is up."""
    return {
        "status":            "ok",
        "proctoring_active": getattr(state, "proctoring_active", False),
        "has_frame":         state.latest_frame is not None,
    }