from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import cv2, time, os

import state
from api.code_routes import router as code_router

app = FastAPI()
templates = Jinja2Templates(directory="templates")

os.makedirs("outputs/evidence", exist_ok=True)
app.mount("/evidence", StaticFiles(directory="outputs/evidence"), name="evidence")
app.include_router(code_router)

# ── Video stream ──────────────────────────────────────────────────────────
def generate_frames():
    while True:
        frame = state.latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame')

# ── Pages ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/exam", response_class=HTMLResponse)
def exam_page(request: Request):
    return templates.TemplateResponse("exam.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

# ── Analytics ─────────────────────────────────────────────────────────────
@app.get("/analytics")
def get_analytics():
    if state.risk_agent is None:
        return {"status": "warming_up"}
    violations = []
    for v in (state.violation_agent.violations if state.violation_agent else []):
        v_copy = dict(v)
        if v_copy.get("evidence"):
            v_copy["evidence_url"] = f"/evidence/{os.path.basename(v_copy['evidence'])}"
        violations.append(v_copy)
    return {
        "suspicion_score":  state.risk_agent.suspicion_score,
        "trust_score":      state.risk_agent.get_trust_score(),
        "risk":             state.risk_agent.get_risk_level(),
        "violations":       violations,
        "timeline":         state.risk_agent.timeline,
        "avg_attention":    0,
        "alert_active":     state.risk_agent.alert_active,
        "latest_alert":     state.risk_agent.get_latest_alert(),
        "breach_count":     state.risk_agent.cutoff_breach_count,
        "tab_switch_count": state.risk_agent.tab_switch_count,
        "test_terminated":  state.risk_agent.test_terminated,
        "active_warning":   state.risk_agent.active_warning,
    }

@app.post("/tab-switch")
def log_tab_switch():
    if state.risk_agent is None:
        return {"status": "no_agent"}
    if state.violation_agent and state.latest_frame is not None:
        state.violation_agent.log_violation("tab_switched", state.latest_frame)
    state.risk_agent.update_risk("tab_switched")
    return {
        "tab_switch_count": state.risk_agent.tab_switch_count,
        "test_terminated":  state.risk_agent.test_terminated,
    }

@app.get("/alert/status")
def alert_status():
    if state.risk_agent is None:
        return {"alert_active": False}
    return {
        "alert_active":    state.risk_agent.alert_active,
        "trust_score":     state.risk_agent.get_trust_score(),
        "suspicion":       state.risk_agent.suspicion_score,
        "breach_count":    state.risk_agent.cutoff_breach_count,
        "latest_alert":    state.risk_agent.get_latest_alert(),
        "test_terminated": state.risk_agent.test_terminated,
    }

@app.post("/alert/dismiss")
def dismiss_alert():
    if state.risk_agent:
        state.risk_agent.alert_active = False
    return {"status": "dismissed"}