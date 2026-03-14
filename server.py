import base64
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import cv2, time, os
import threading
from fastapi.middleware.cors import CORSMiddleware
import state
from main import run_proctoring

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all domains
    allow_credentials=True,
    allow_methods=["*"],      # allow GET, POST, PUT, DELETE etc
    allow_headers=["*"],      # allow all headers
)


os.makedirs("outputs/evidence", exist_ok=True)
app.mount("/evidence", StaticFiles(directory="outputs/evidence"), name="evidence")


# ── Video stream ──────────────────────────────────────────────────────────


@app.on_event("startup")
def start_proctoring():
    print("Starting AI Proctoring Thread...")
    proctor_thread = threading.Thread(target=run_proctoring, daemon=True)
    proctor_thread.start()


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




@app.post("/video/frame")
async def receive_frame(request: Request):
    state.proctoring_active = True  


    data = await request.json()

    image = data["image"]
    assessment_id = data["assessment_id"]
    email_id = data["email_id"]

    # decode base64 image
    img_bytes = base64.b64decode(image.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # store frame
    state.latest_frame = frame

    # optionally store metadata
    state.assessment_id = assessment_id
    state.email_id = email_id

    return {
        "status": "frame received",
        "assessment_id": assessment_id,
        "email_id": email_id
    }

@app.post("/stop")
def stop_proctoring():
    state.proctoring_active = False
    return {"status": "proctoring stopped"}