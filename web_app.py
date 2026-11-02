"""
YOLO Real-Time Detection & Tracking Web Dashboard Server
Powered by FastAPI, OpenCV, and modern YOLO with GPU acceleration.
Architecture: Dedicated Singleton Worker Thread for 100% thread-safe MJPEG streaming.
"""
import io
import os
import shutil
import sys
import threading
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from yolo_detector import YOLODetector

app = FastAPI(title="NeuralVision YOLO AI Dashboard", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
STATIC_DIR = os.path.abspath("static")
UPLOAD_DIR = os.path.abspath("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def create_placeholder_jpeg(text="INITIALIZING STREAM..."):
    """Creates a sleek fallback image in case stream hasn't produced a frame yet."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Dark gradient background
    for y in range(720):
        img[y, :] = (int(16 + y * 0.02), int(14 + y * 0.02), int(22 + y * 0.02))
    cv2.putText(
        img,
        text,
        (430, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "NEURALVISION AI ENGINE",
        (485, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (130, 140, 170),
        1,
        cv2.LINE_AA,
    )
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes()


class StreamManager:
    """
    Singleton Stream Manager running a dedicated worker thread.
    Guarantees that cv2.VideoCapture and YOLODetector are only accessed by a single thread,
    preventing any libavcodec / pthread assertion crashes when multiple browser clients connect.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_condition = threading.Condition()

        # Stream configuration
        self.source_type = "video"  # "video", "webcam", "image"
        self.source_path = "uk.mp4"
        self.next_source = None  # Tuple of (source_type, source_path) requested
        self.model_name = "yolo11n"
        self.next_model = None
        self.conf_threshold = 0.35
        self.iou_threshold = 0.45
        self.enable_tracking = True
        self.loop = True
        self.paused = False

        # Live telemetry metrics
        self.fps = 0.0
        self.latency_ms = 0.0
        self.current_detections = []
        self.class_counts = {}
        self.gpu_device_name = "Initializing..."

        # Frame buffers
        self.latest_jpeg = create_placeholder_jpeg("STARTING NEURAL VISION...")
        self.annotated_frame = None

        # Detector
        self.detector = None
        self._init_detector()

        # Start dedicated background capture & inference worker
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop, name="YOLO-StreamWorker", daemon=True
        )
        self.worker_thread.start()

    def _init_detector(self):
        try:
            self.detector = YOLODetector(
                model=self.model_name,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )
            self.gpu_device_name = getattr(self.detector, "device_name", "CPU")
        except Exception as e:
            print(f"[Detector Init Error]: {e}")
            self.gpu_device_name = "CPU"

    def change_source(self, source_type: str, source_path: str):
        with self.lock:
            self.next_source = (source_type, source_path)
            print(f"[StreamManager] Source change requested -> {source_type}: {source_path}")

    def update_config(self, model=None, conf=None, iou=None, tracking=None, loop=None):
        with self.lock:
            if model and model != self.model_name:
                self.next_model = model
            if conf is not None:
                self.conf_threshold = float(conf)
                if self.detector:
                    self.detector.conf_threshold = self.conf_threshold
            if iou is not None:
                self.iou_threshold = float(iou)
                if self.detector:
                    self.detector.iou_threshold = self.iou_threshold
            if tracking is not None:
                self.enable_tracking = bool(tracking)
            if loop is not None:
                self.loop = bool(loop)

    def get_telemetry_dict(self):
        with self.lock:
            return {
                "fps": self.fps,
                "latency_ms": self.latency_ms,
                "active_model": self.model_name,
                "hardware": self.gpu_device_name,
                "source_type": self.source_type,
                "source_name": os.path.basename(self.source_path),
                "object_count": len(self.current_detections),
                "class_counts": self.class_counts,
                "paused": self.paused,
                "tracking": self.enable_tracking,
                "confidence": self.conf_threshold,
                "iou": self.iou_threshold,
                "detections": [
                    {
                        "label": d["label"],
                        "track_id": d.get("track_id"),
                        "confidence": round(d["confidence"] * 100, 1),
                        "box": d["box"],
                    }
                    for d in self.current_detections[:25]
                ],
            }

    def _worker_loop(self):
        """
        Single dedicated thread responsible for all video capture, decoding,
        and YOLO inference. Completely isolates OpenCV and PyTorch from HTTP request threads.
        """
        cap = None
        static_img = None
        cached_annotated = None
        cached_jpeg = None
        cached_detections = []
        cached_counts = {}

        def open_source(stype, spath):
            nonlocal cap, static_img, cached_annotated, cached_jpeg, cached_detections, cached_counts
            cached_annotated = None
            cached_jpeg = None
            cached_detections = []
            cached_counts = {}
            if self.detector:
                self.detector.reset_tracker()
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            static_img = None

            if not os.path.isabs(spath) and stype != "webcam":
                spath = os.path.abspath(spath)

            print(f"[Worker] Opening source: {stype} ({spath})")
            if stype == "webcam":
                backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
                idx = int(spath) if str(spath).isdigit() else 0
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(idx)
            elif stype == "video":
                if os.path.exists(spath):
                    cap = cv2.VideoCapture(spath)
                else:
                    print(f"[Worker Warning] Video file does not exist: {spath}")
            elif stype == "image":
                if os.path.exists(spath):
                    static_img = cv2.imread(spath)
                    print(f"[Worker] Successfully loaded image {spath}, shape: {static_img.shape if static_img is not None else 'None'}")
                else:
                    print(f"[Worker Warning] Image file does not exist: {spath}")

        open_source(self.source_type, self.source_path)

        frame_counter = 0
        t_fps_start = time.time()

        while self.running:
            # Check for source change or model switch
            with self.lock:
                if self.next_source is not None:
                    new_type, new_path = self.next_source
                    self.next_source = None
                    self.source_type = new_type
                    self.source_path = new_path
                    open_source(new_type, new_path)

                if self.next_model is not None:
                    new_model = self.next_model
                    self.next_model = None
                    self.model_name = new_model
                    self._init_detector()
                    cached_jpeg = None  # Force re-infer

                paused = self.paused
                detector = self.detector
                tracking = self.enable_tracking
                loop = self.loop
                current_type = self.source_type

            # If stream is paused, broadcast current frame at steady 25 FPS
            if paused:
                time.sleep(0.04)
                continue

            frame = None

            # Handle Static Image
            if current_type == "image":
                if static_img is not None:
                    if cached_jpeg is not None:
                        # Stream the cached annotated JPEG smoothly at 30 FPS without re-inferencing
                        with self.lock:
                            self.latest_jpeg = cached_jpeg
                            self.annotated_frame = cached_annotated
                            self.current_detections = cached_detections
                            self.class_counts = cached_counts
                            self.fps = 30.0
                        with self.frame_condition:
                            self.frame_condition.notify_all()
                        time.sleep(0.033)
                        continue
                    frame = static_img.copy()
                else:
                    time.sleep(0.05)
                    continue

            # Handle Video / Webcam
            elif cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    if loop and current_type == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if detector:
                            detector.reset_tracker()
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.05)
                            continue
                    else:
                        time.sleep(0.05)
                        continue
            else:
                time.sleep(0.05)
                continue

            if frame is None:
                time.sleep(0.03)
                continue

            # Downscale frame for inference & web streaming if resolution is very high (e.g. 4K)
            h, w = frame.shape[:2]
            if w > 1280 or h > 720:
                scale = min(1280.0 / w, 720.0 / h)
                dw = int(w * scale)
                dh = int(h * scale)
                frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)

            # YOLO Inference
            try:
                # Do NOT run motion tracking on static images
                run_track = tracking if current_type != "image" else False
                detections, latency_ms = detector.detect(frame, track=run_track)
                annotated = detector.draw_detections(
                    frame,
                    detections,
                    fps=self.fps if current_type != "image" else 30.0,
                    inference_time_ms=latency_ms,
                    show_hud=False,
                )

                # JPEG encoding
                _, jpeg_buf = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                jpeg_bytes = jpeg_buf.tobytes()

                # Calculate FPS
                frame_counter += 1
                now = time.time()
                dt = now - t_fps_start
                if dt >= 0.5:
                    self.fps = round(frame_counter / dt, 1)
                    frame_counter = 0
                    t_fps_start = now

                # Class breakdown
                counts = {}
                for d in detections:
                    lbl = d["label"]
                    counts[lbl] = counts.get(lbl, 0) + 1

                # Update shared state
                with self.lock:
                    self.latency_ms = round(latency_ms, 1)
                    self.current_detections = detections
                    self.class_counts = counts
                    self.annotated_frame = annotated
                    self.latest_jpeg = jpeg_bytes

                if current_type == "image":
                    cached_jpeg = jpeg_bytes
                    cached_annotated = annotated
                    cached_detections = detections
                    cached_counts = counts

                # Notify all connected client stream generators
                with self.frame_condition:
                    self.frame_condition.notify_all()

                # For video playback, apply gentle pacing so video plays at natural speed
                if current_type == "video":
                    # Target ~30 FPS (approx 33ms per frame)
                    remaining = 0.030 - (latency_ms / 1000.0)
                    if remaining > 0.005:
                        time.sleep(remaining)

            except Exception as e:
                print(f"[Worker Inference Error]: {e}")
                time.sleep(0.05)


stream_manager = StreamManager()


def generate_frames():
    """
    Client stream generator.
    Passively broadcasts the latest frame from the StreamManager worker thread.
    Supports unlimited concurrent clients without duplicating inference or touching OpenCV.
    """
    while True:
        frame_bytes = None
        with stream_manager.frame_condition:
            # Wait for next frame (timeout to maintain heartbeat)
            stream_manager.frame_condition.wait(timeout=0.08)
            frame_bytes = stream_manager.latest_jpeg

        if frame_bytes is None:
            time.sleep(0.02)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/telemetry")
def get_telemetry():
    """Real-time statistics endpoint for dashboard UI widgets."""
    return stream_manager.get_telemetry_dict()


class ConfigPayload(BaseModel):
    model: str = None
    confidence: float = None
    iou: float = None
    tracking: bool = None
    loop: bool = None
    paused: bool = None


@app.post("/api/config")
def update_config(payload: ConfigPayload):
    if payload.paused is not None:
        stream_manager.paused = payload.paused

    stream_manager.update_config(
        model=payload.model,
        conf=payload.confidence,
        iou=payload.iou,
        tracking=payload.tracking,
        loop=payload.loop,
    )
    return {"status": "success", "message": "Configuration updated"}


class SourcePayload(BaseModel):
    type: str  # "video", "webcam", "image"
    path: str


@app.post("/api/source")
def set_source(payload: SourcePayload):
    target = payload.path
    if payload.type != "webcam" and not os.path.isabs(target):
        target = os.path.abspath(target)
    stream_manager.change_source(payload.type, target)
    return {"status": "success", "source": target}


@app.post("/api/reset_tracks")
def reset_tracks():
    """Resets tracking IDs back to #1."""
    if stream_manager.detector:
        stream_manager.detector.reset_tracker()
    return {"status": "success", "message": "Tracking IDs reset to #1"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles custom video or image uploads directly from browser."""
    filename = file.filename
    clean_name = os.path.basename(filename)
    dest_path = os.path.join(UPLOAD_DIR, clean_name)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = os.path.splitext(clean_name)[1].lower()
    is_img = ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    stype = "image" if is_img else "video"

    stream_manager.change_source(stype, dest_path)
    return {
        "status": "success",
        "filename": clean_name,
        "type": stype,
        "path": dest_path,
    }


@app.get("/api/snapshot")
def take_snapshot():
    """Captures and downloads current processed frame as JPEG image."""
    with stream_manager.lock:
        if stream_manager.annotated_frame is None:
            raise HTTPException(status_code=404, detail="No frame available")
        _, img_encoded = cv2.imencode(".jpg", stream_manager.annotated_frame)
        data = img_encoded.tobytes()

    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=snapshot_{int(time.time())}.jpg"},
    )


# Serve Static Assets & Main Web App
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>YOLO Dashboard: static/index.html not found</h1>"


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 65)
    print("      NEURALVISION // YOLO REAL-TIME AI DASHBOARD")
    print("=" * 65)
    print(" -> Dashboard URL: http://localhost:8000")
    print(" -> GPU Device:    NVIDIA GeForce RTX 2050")
    print(" -> Engine:        Dedicated Worker Thread (Zero Crash MJPEG)")
    print("=" * 65 + "\n")
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
