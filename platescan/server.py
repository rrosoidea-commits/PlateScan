"""
PlateScan - ALPR dashboard backend
==================================
A small FastAPI server that runs the license plate recognition pipeline
and serves the dashboard page (static/index.html).

Run from a terminal:
    python platescan/server.py
then open http://127.0.0.1:8000 in your browser.

Pipeline for every image (camera frame or upload):
  1. Vehicle detection   - YOLO26n pretrained on COCO (car, motorcycle, bus, truck)
  2. Plate localization  - our trained YOLO26n plate detector (best.pt)
  3. Preprocessing       - crop + grayscale / CLAHE / threshold (shown for inspection)
  4. OCR                 - fast-plate-ocr (cct-xs-v2-global-model) on the color crop
  5. Format check + log  - Philippine plate format check, saved to SQLite

Everything runs locally; no image leaves this computer.
"""

import base64
import csv
import io
import json
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DB_PATH = APP_DIR / "platescan.db"

ROOT = APP_DIR.parent                                                # the repository folder
PROJECT_DIR = ROOT / "plate_detection"                               # evaluation outputs (optional)
PLATE_WEIGHTS = ROOT / "models" / "yolo26n_640_e50_best.pt"          # our trained detector
VEHICLE_WEIGHTS = APP_DIR / "weights" / "yolo26n_coco.pt"           # COCO-pretrained YOLO26n
DETECTOR_METRICS = PROJECT_DIR / "outputs" / "yolo26n_640_e50_metrics.json"
REPORT_NUMBERS = PROJECT_DIR / "report_figures" / "report_numbers.json"
OCR_RESULTS = PROJECT_DIR / "ocr_outputs" / "ocr_results.csv"
MANUAL_CHECK = PROJECT_DIR / "ocr_outputs" / "manual_check.csv"

OCR_MODEL = "cct-xs-v2-global-model"
PLATE_CONF = 0.40       # detector threshold (0.4-0.5 gave the best F1 on the test split)
VEHICLE_CONF = 0.30
PAD = 0.05              # margin around each plate crop (5% of its size)
REVIEW_OCR_CONF = 0.80  # reads below this OCR confidence are flagged for review

# COCO class ids -> vehicle names
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Philippine plate formats (letters/digits only, spaces removed).
# Verify against current LTO rules before relying on them.
PH_FORMATS = [
    (re.compile(r"^[A-Z]{3}\d{4}$"), "ABC 1234", "private / for-hire (current series)"),
    (re.compile(r"^[A-Z]{3}\d{3}$"), "ABC 123", "older series"),
    (re.compile(r"^\d{3}[A-Z]{3}$"), "123 ABC", "motorcycle (older series)"),
]

# OCR accuracy measured during development on 23 hand-labeled test crops
# (used until manual_check.csv is filled in).
OCR_DEV_SAMPLE = {"plates": 23, "exact": 20, "char_accuracy": 0.980}
# Full-pipeline speed measured on the 593 test images (AI_PlateNumber_OCR.py)
PIPELINE_TEST_SPEED = {"ms_per_image": 23.8, "fps": 42.0}


# ---------------------------------------------------------------------------
# Models (loaded once at startup)
# ---------------------------------------------------------------------------
class Models:
    def __init__(self):
        import onnxruntime
        import torch
        from fast_plate_ocr import LicensePlateRecognizer
        from ultralytics import YOLO

        onnxruntime.set_default_logger_severity(3)  # hide harmless ONNX warnings
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        if not PLATE_WEIGHTS.exists():
            raise FileNotFoundError(f"Plate detector not found: {PLATE_WEIGHTS}")
        self.plate = YOLO(str(PLATE_WEIGHTS))
        self.vehicle = YOLO(str(VEHICLE_WEIGHTS)) if VEHICLE_WEIGHTS.exists() else YOLO("yolo26n.pt")
        self.ocr = LicensePlateRecognizer(OCR_MODEL, device="cpu")
        self.lock = threading.Lock()  # one image at a time on the GPU

        # Warm-up so the first real request is fast
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.plate.predict(dummy, device=self.device, verbose=False)
        self.vehicle.predict(dummy, device=self.device, verbose=False)
        self.ocr.run(np.zeros((64, 128, 3), dtype=np.uint8))


MODELS = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR image, with phone EXIF rotation applied."""
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image file.")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def to_b64(img: np.ndarray, ext=".jpg", quality=85) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    mime = "jpeg" if ext == ".jpg" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf).decode() if ok else ""


def check_format(text: str):
    """Return (is_valid, pretty_text, description) for a Philippine plate format."""
    for pattern, example, desc in PH_FORMATS:
        if pattern.match(text):
            m = re.match(r"^([A-Z]+)(\d+)$", text) or re.match(r"^(\d+)([A-Z]+)$", text)
            pretty = f"{m.group(1)} {m.group(2)}" if m else text
            return True, pretty, f"{example} · {desc}"
    return False, text, "not a Philippine format"


def preprocess_views(crop: np.ndarray):
    """The four inspection views shown in the dashboard."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    big = lambda im: cv2.resize(im, (max(1, int(im.shape[1] * 96 / im.shape[0])), 96), interpolation=cv2.INTER_CUBIC)
    return [to_b64(big(crop), ".png"), to_b64(big(gray), ".png"), to_b64(big(clahe), ".png"), to_b64(big(thresh), ".png")]


def box_contains(outer, x, y):
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


# ---------------------------------------------------------------------------
# The recognition pipeline
# ---------------------------------------------------------------------------
def recognize(img: np.ndarray):
    h, w = img.shape[:2]
    t = {}

    # 1. Vehicle detection (COCO classes)
    t0 = time.perf_counter()
    vr = MODELS.vehicle.predict(img, imgsz=640, conf=VEHICLE_CONF, device=MODELS.device, verbose=False,
                                classes=list(VEHICLE_CLASSES))[0]
    vehicles = [{"label": VEHICLE_CLASSES[int(c)], "conf": float(s), "box": [int(v) for v in b]}
                for b, c, s in zip(vr.boxes.xyxy.cpu().numpy(), vr.boxes.cls.cpu().numpy(), vr.boxes.conf.cpu().numpy())]
    t["vehicle"] = (time.perf_counter() - t0) * 1000

    # 2. Plate localization (our trained detector, on the full image)
    t0 = time.perf_counter()
    pr = MODELS.plate.predict(img, imgsz=640, conf=PLATE_CONF, device=MODELS.device, verbose=False)[0]
    t["plate"] = (time.perf_counter() - t0) * 1000

    plates = []
    t_pre = t_ocr = 0.0
    for box, conf in zip(pr.boxes.xyxy.cpu().numpy(), pr.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        crop = img[y1:y2, x1:x2]

        # 3. Preprocessing (inspection views; the OCR model reads the color crop)
        t0 = time.perf_counter()
        views = preprocess_views(crop)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t_pre += (time.perf_counter() - t0) * 1000

        # 4. OCR
        t0 = time.perf_counter()
        pred = MODELS.ocr.run(rgb, return_confidence=True)[0]
        t_ocr += (time.perf_counter() - t0) * 1000
        text = "".join(ch for ch in pred.plate.upper() if ch.isalnum())
        ocr_conf = float(pred.char_probs[:len(text)].mean()) if text else 0.0

        # 5. Format check
        valid, pretty, fmt_desc = check_format(text)

        # Which vehicle is this plate on? (vehicle box containing the plate center)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        owner = max((v for v in vehicles if box_contains(v["box"], cx, cy)),
                    key=lambda v: v["conf"], default=None)

        reasons = []
        if not text:
            reasons.append("no text read")
        elif ocr_conf < REVIEW_OCR_CONF:
            reasons.append("low OCR confidence")
        if text and not valid:
            reasons.append("format")
        plates.append({
            "plate": pretty if text else "",
            "plate_raw": text,
            "ocr_conf": round(ocr_conf, 3),
            "plate_conf": round(float(conf), 3),
            "plate_box": [x1, y1, x2, y2],
            "plate_height_px": int(y2 - y1),
            "region": getattr(pred, "region", None),
            "region_conf": round(float(getattr(pred, "region_prob", 0) or 0), 3),
            "format_valid": valid,
            "format": fmt_desc,
            "vehicle": owner["label"] if owner else None,
            "vehicle_conf": round(owner["conf"], 3) if owner else None,
            "vehicle_box": owner["box"] if owner else None,
            "status": "Review" if reasons else "OK",
            "review_reasons": reasons,
            "preprocess": views,
        })
    t["preprocess"] = t_pre
    t["ocr"] = t_ocr

    # Annotated image for the dashboard
    vis = img.copy()
    thick = max(2, w // 400)
    for v in vehicles:
        b = v["box"]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (214, 120, 42), thick)  # blue-ish (BGR)
        cv2.putText(vis, f"{v['label']} {v['conf']:.2f}", (b[0] + 4, b[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (214, 120, 42), 2)
    for p in plates:
        b = p["plate_box"]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (54, 186, 245), thick + 1)  # amber (BGR)
        label = p["plate"] or "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty = max(b[1] - 8, th + 8)
        cv2.rectangle(vis, (b[0], ty - th - 8), (b[0] + tw + 10, ty + 4), (54, 186, 245), -1)
        cv2.putText(vis, label, (b[0] + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    plates.sort(key=lambda p: (p["plate_conf"] * (p["ocr_conf"] or 0.01)), reverse=True)
    return plates, vehicles, t, vis


# ---------------------------------------------------------------------------
# SQLite vehicle log
# ---------------------------------------------------------------------------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS vehicle_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plate TEXT,
            vehicle TEXT,
            ocr_conf REAL,
            det_conf REAL,
            source TEXT,
            status TEXT,
            reasons TEXT,
            session TEXT
        )""")


SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Model performance numbers for the dashboard
# ---------------------------------------------------------------------------
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def performance_numbers():
    out = {"detection": {}, "ocr": {}, "system": {}}
    if DETECTOR_METRICS.exists():
        m = json.loads(DETECTOR_METRICS.read_text())
        tm = m["test_metrics"]
        out["detection"] = {"mAP50": tm["mAP50"], "mAP50_95": tm["mAP50-95"],
                            "precision": tm["precision"], "recall": tm["recall"]}
        out["system"]["test_images"] = m["dataset"]["test"]["images"]
    recall_025 = None
    if REPORT_NUMBERS.exists():
        rn = json.loads(REPORT_NUMBERS.read_text())
        for s in rn.get("threshold_sweep", []):
            if abs(s["conf"] - 0.25) < 1e-6:
                recall_025 = s["recall"]

    # OCR accuracy: manual_check.csv if filled in, otherwise the 23-crop development sample
    current = {}
    if OCR_RESULTS.exists():
        with open(OCR_RESULTS, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        current = {r["crop_file"]: r["text"] for r in rows}
        texts = [r["text"] for r in rows if r["text"]]
        out["ocr"]["valid_format_rate"] = (sum(check_format(t)[0] for t in texts) / len(texts)) if texts else None
    source, exact, chars = "23-plate development sample", OCR_DEV_SAMPLE["exact"] / OCR_DEV_SAMPLE["plates"], OCR_DEV_SAMPLE["char_accuracy"]
    if MANUAL_CHECK.exists():
        with open(MANUAL_CHECK, newline="", encoding="utf-8") as f:
            filled = [r for r in csv.DictReader(f) if r["correct_text"].strip() not in ("", "-")]
        if filled:
            ex = errs = tot = 0
            for r in filled:
                ocr = "".join(c for c in current.get(r["crop_file"], r["ocr_text"]).upper() if c.isalnum())
                truth = "".join(c for c in r["correct_text"].upper() if c.isalnum())
                d = edit_distance(ocr, truth)
                ex += d == 0; errs += d; tot += len(truth)
            source, exact, chars = f"manual check ({len(filled)} plates)", ex / len(filled), 1 - errs / max(1, tot)
    out["ocr"].update({"plate_accuracy": exact, "char_accuracy": chars, "cer": 1 - chars, "source": source})
    if recall_025 is not None:
        out["system"]["end_to_end"] = recall_025 * exact
    out["system"].update({"latency_ms": PIPELINE_TEST_SPEED["ms_per_image"], "fps": PIPELINE_TEST_SPEED["fps"]})
    return out


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    """Load the models once when the server starts."""
    global MODELS
    init_db()
    print("Loading models...")
    MODELS = Models()
    print(f"Models ready on {MODELS.device_name}. Open http://127.0.0.1:8000")
    yield


app = FastAPI(title="PlateScan", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": MODELS.device_name if MODELS else None,
            "models": {"plate": PLATE_WEIGHTS.name, "vehicle": VEHICLE_WEIGHTS.name, "ocr": OCR_MODEL},
            "plate_conf": PLATE_CONF, "review_ocr_conf": REVIEW_OCR_CONF}


@app.get("/metrics")
def metrics():
    return performance_numbers()


@app.post("/predict")
def predict(image: UploadFile = File(...), source: str = Form("upload")):
    img = load_image(image.file.read())
    t_start = time.perf_counter()
    with MODELS.lock:
        plates, vehicles, timings, vis = recognize(img)
    total = (time.perf_counter() - t_start) * 1000

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db() as con:
        for p in plates:
            con.execute("INSERT INTO vehicle_log (timestamp, plate, vehicle, ocr_conf, det_conf, source, status, reasons, session)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (ts, p["plate"], p["vehicle"], p["ocr_conf"], p["plate_conf"], source,
                         p["status"], ", ".join(p["review_reasons"]), SESSION_ID))

    best = plates[0] if plates else None
    return {
        # main reading (same fields as the API sketch in the mockup)
        "plate": best["plate"] if best else None,
        "ocr_conf": best["ocr_conf"] if best else None,
        "plate_conf": best["plate_conf"] if best else None,
        "vehicle": best["vehicle"] if best else None,
        "vehicle_conf": best["vehicle_conf"] if best else None,
        "vehicle_box": best["vehicle_box"] if best else None,
        "plate_box": best["plate_box"] if best else None,
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "total_ms": round(total, 1),
        # extra detail for the dashboard
        "timestamp": ts,
        "source": source,
        "image_size": [img.shape[1], img.shape[0]],
        "plates": plates,
        "vehicles": vehicles,
        "annotated": to_b64(vis if max(vis.shape[:2]) <= 1280 else
                            cv2.resize(vis, None, fx=1280 / max(vis.shape[:2]), fy=1280 / max(vis.shape[:2]))),
    }


@app.get("/log")
def get_log(search: str = "", limit: int = 200):
    q = "%" + "".join(c for c in search.upper() if c.isalnum() or c == " ") + "%"
    with db() as con:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM vehicle_log WHERE REPLACE(plate,' ','') LIKE REPLACE(?,' ','') ORDER BY id DESC LIMIT ?",
            (q, limit))]
        s = con.execute("SELECT COUNT(*) n, AVG(ocr_conf) avg_conf, SUM(status='Review') flagged "
                        "FROM vehicle_log WHERE session=?", (SESSION_ID,)).fetchone()
    return {"rows": rows, "session": {"logged": s["n"], "avg_ocr_conf": s["avg_conf"], "flagged": s["flagged"] or 0}}


@app.get("/log.csv")
def export_csv():
    with db() as con:
        rows = con.execute("SELECT id, timestamp, plate, vehicle, ocr_conf, det_conf, source, status, reasons "
                           "FROM vehicle_log ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "timestamp", "plate", "vehicle", "ocr_conf", "det_conf", "source", "status", "reasons"])
    w.writerows([tuple(r) for r in rows])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=platescan_log.csv"})


@app.delete("/log")
def clear_log():
    with db() as con:
        con.execute("DELETE FROM vehicle_log")
    return {"cleared": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
