"""
PlateScan - shared recognition pipeline
=======================================
Used by both front ends:
  - platescan/server.py   (FastAPI + HTML dashboard, local)
  - streamlit_app.py      (Streamlit app, local or Streamlit Community Cloud)

Pipeline for every image:
  1. Vehicle detection   - YOLO26n pretrained on COCO (car, motorcycle, bus, truck)
  2. Plate localization  - our trained YOLO26n plate detector (models/...best.pt)
  3. Preprocessing       - crop + grayscale / CLAHE / threshold (shown for inspection)
  4. OCR                 - fast-plate-ocr (cct-xs-v2-global-model) on the color crop
  5. Format check + log  - Philippine plate format check, saved to SQLite
"""

import csv
import io
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent                                          # the repository folder
PLATE_WEIGHTS = ROOT / "models" / "yolo26n_640_e50_best.pt"    # our trained detector
VEHICLE_WEIGHTS = APP_DIR / "weights" / "yolo26n_coco.pt"      # COCO YOLO26n (downloaded if missing)

# Evaluation outputs (only present after running the training/OCR scripts locally)
PROJECT_DIR = ROOT / "plate_detection"
DETECTOR_METRICS = PROJECT_DIR / "outputs" / "yolo26n_640_e50_metrics.json"
REPORT_NUMBERS = PROJECT_DIR / "report_figures" / "report_numbers.json"
OCR_RESULTS = PROJECT_DIR / "ocr_outputs" / "ocr_results.csv"
MANUAL_CHECK = PROJECT_DIR / "ocr_outputs" / "manual_check.csv"

OCR_MODEL = "cct-xs-v2-global-model"
PLATE_CONF = 0.40       # detector threshold (0.4-0.5 gave the best F1 on the test split)
VEHICLE_CONF = 0.30
PAD = 0.05              # margin around each plate crop (5% of its size)
REVIEW_OCR_CONF = 0.80  # reads below this OCR confidence are flagged for review

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO class ids

# Philippine plate formats (letters/digits only, spaces removed).
# Verify against current LTO rules before relying on them.
PH_FORMATS = [
    (re.compile(r"^[A-Z]{3}\d{4}$"), "ABC 1234", "private / for-hire (current series)"),
    (re.compile(r"^[A-Z]{3}\d{3}$"), "ABC 123", "older series"),
    (re.compile(r"^\d{3}[A-Z]{3}$"), "123 ABC", "motorcycle (older series)"),
]

# Results measured during the project, used when the evaluation output files are not
# available (e.g. on Streamlit Cloud, where plate_detection/ is not in the repository).
EVALUATION_DEFAULTS = {
    "detection": {"mAP50": 0.9463, "mAP50_95": 0.7919, "precision": 0.985, "recall": 0.926},
    "recall_at_025": 0.9408,          # detector recall at conf 0.25, IoU 0.5 (test split)
    "test_images": 593,
    "ocr_dev_sample": {"plates": 23, "exact": 20, "char_accuracy": 0.980},
    "valid_format_rate": 0.147,       # share of test-set reads matching a PH format
    "pipeline_speed": {"ms_per_image": 23.8, "fps": 42.0, "device": "NVIDIA RTX 2050"},
}


# ---------------------------------------------------------------------------
# Models (load once, reuse for every image)
# ---------------------------------------------------------------------------
class Models:
    def __init__(self, device=None):
        import onnxruntime
        import torch
        from fast_plate_ocr import LicensePlateRecognizer
        from ultralytics import YOLO

        onnxruntime.set_default_logger_severity(3)  # hide harmless ONNX warnings
        forced = device or os.environ.get("PLATESCAN_DEVICE")  # e.g. "cpu" to test cloud speed
        if forced:
            self.device = forced
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        self.device_name = (torch.cuda.get_device_name(0) if self.device != "cpu" else "CPU")
        if not PLATE_WEIGHTS.exists():
            raise FileNotFoundError(f"Plate detector not found: {PLATE_WEIGHTS}")
        self.plate = YOLO(str(PLATE_WEIGHTS))
        self.vehicle = YOLO(str(VEHICLE_WEIGHTS)) if VEHICLE_WEIGHTS.exists() else YOLO("yolo26n.pt")
        self.ocr = LicensePlateRecognizer(OCR_MODEL, device="cpu")
        self.lock = threading.Lock()  # one image at a time

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)  # warm-up
        self.plate.predict(dummy, device=self.device, verbose=False)
        self.vehicle.predict(dummy, device=self.device, verbose=False)
        self.ocr.run(np.zeros((64, 128, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image(data: bytes) -> np.ndarray:
    """Image bytes -> BGR array, with phone EXIF rotation applied. Raises ValueError."""
    try:
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        raise ValueError("Could not read the image file.") from e
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def check_format(text: str):
    """Return (is_valid, pretty_text, description) for a Philippine plate format."""
    for pattern, example, desc in PH_FORMATS:
        if pattern.match(text):
            m = re.match(r"^([A-Z]+)(\d+)$", text) or re.match(r"^(\d+)([A-Z]+)$", text)
            pretty = f"{m.group(1)} {m.group(2)}" if m else text
            return True, pretty, f"{example} · {desc}"
    return False, text, "not a Philippine format"


def preprocess_views(crop: np.ndarray):
    """Four inspection views of a plate crop: color crop, grayscale, CLAHE, threshold (96 px tall)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def big(im):
        return cv2.resize(im, (max(1, int(im.shape[1] * 96 / im.shape[0])), 96), interpolation=cv2.INTER_CUBIC)
    return [big(crop), big(gray), big(clahe), big(thresh)]


def box_contains(outer, x, y):
    return outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]


# ---------------------------------------------------------------------------
# The recognition pipeline
# ---------------------------------------------------------------------------
def recognize(models: Models, img: np.ndarray):
    """Run the 5-step pipeline. Returns (plates, vehicles, timings_ms, annotated_bgr).

    plates are sorted best first; each has: plate, plate_raw, ocr_conf, plate_conf, plate_box,
    region, region_conf, format_valid, format, vehicle, vehicle_conf, vehicle_box, status,
    review_reasons, views (4 numpy images).
    """
    with models.lock:
        h, w = img.shape[:2]
        t = {}

        # 1. Vehicle detection (COCO classes)
        t0 = time.perf_counter()
        vr = models.vehicle.predict(img, imgsz=640, conf=VEHICLE_CONF, device=models.device, verbose=False,
                                    classes=list(VEHICLE_CLASSES))[0]
        vehicles = [{"label": VEHICLE_CLASSES[int(c)], "conf": float(s), "box": [int(v) for v in b]}
                    for b, c, s in zip(vr.boxes.xyxy.cpu().numpy(), vr.boxes.cls.cpu().numpy(),
                                       vr.boxes.conf.cpu().numpy())]
        t["vehicle"] = (time.perf_counter() - t0) * 1000

        # 2. Plate localization (our trained detector, on the full image)
        t0 = time.perf_counter()
        pr = models.plate.predict(img, imgsz=640, conf=PLATE_CONF, device=models.device, verbose=False)[0]
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
            pred = models.ocr.run(rgb, return_confidence=True)[0]
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
                "region": getattr(pred, "region", None),
                "region_conf": round(float(getattr(pred, "region_prob", 0) or 0), 3),
                "format_valid": valid,
                "format": fmt_desc,
                "vehicle": owner["label"] if owner else None,
                "vehicle_conf": round(owner["conf"], 3) if owner else None,
                "vehicle_box": owner["box"] if owner else None,
                "status": "Review" if reasons else "OK",
                "review_reasons": reasons,
                "views": views,
            })
        t["preprocess"] = t_pre
        t["ocr"] = t_ocr

    # Annotated image: vehicles in blue, plates in amber with the text read
    vis = img.copy()
    thick = max(2, w // 400)
    for v in vehicles:
        b = v["box"]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (214, 120, 42), thick)
        cv2.putText(vis, f"{v['label']} {v['conf']:.2f}", (b[0] + 4, b[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (214, 120, 42), 2)
    for p in plates:
        b = p["plate_box"]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (54, 186, 245), thick + 1)
        label = p["plate"] or "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty = max(b[1] - 8, th + 8)
        cv2.rectangle(vis, (b[0], ty - th - 8), (b[0] + tw + 10, ty + 4), (54, 186, 245), -1)
        cv2.putText(vis, label, (b[0] + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    plates.sort(key=lambda p: p["plate_conf"] * (p["ocr_conf"] or 0.01), reverse=True)
    return plates, vehicles, t, vis


# ---------------------------------------------------------------------------
# SQLite vehicle log
# ---------------------------------------------------------------------------
LOG_COLUMNS = ["id", "timestamp", "plate", "vehicle", "ocr_conf", "det_conf", "source", "status", "reasons"]


class VehicleLog:
    def __init__(self, path):
        self.path = Path(path)
        with self._db() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS vehicle_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, plate TEXT, vehicle TEXT,
                ocr_conf REAL, det_conf REAL, source TEXT, status TEXT, reasons TEXT, session TEXT)""")

    def _db(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def add(self, plates, timestamp, source, session):
        with self._db() as con:
            for p in plates:
                con.execute("INSERT INTO vehicle_log (timestamp, plate, vehicle, ocr_conf, det_conf, source,"
                            " status, reasons, session) VALUES (?,?,?,?,?,?,?,?,?)",
                            (timestamp, p["plate"], p["vehicle"], p["ocr_conf"], p["plate_conf"], source,
                             p["status"], ", ".join(p["review_reasons"]), session))

    def rows(self, search="", session=None, limit=200):
        q = "%" + "".join(c for c in search.upper() if c.isalnum()) + "%"
        sql = "SELECT * FROM vehicle_log WHERE REPLACE(plate,' ','') LIKE ?"
        args = [q]
        if session:
            sql += " AND session = ?"
            args.append(session)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._db() as con:
            return [dict(r) for r in con.execute(sql, args)]

    def stats(self, session):
        with self._db() as con:
            s = con.execute("SELECT COUNT(*) n, AVG(ocr_conf) avg_conf, SUM(status='Review') flagged "
                            "FROM vehicle_log WHERE session=?", (session,)).fetchone()
        return {"logged": s["n"], "avg_ocr_conf": s["avg_conf"], "flagged": s["flagged"] or 0}

    def to_csv(self, session=None):
        sql = "SELECT " + ", ".join(LOG_COLUMNS) + " FROM vehicle_log"
        args = []
        if session:
            sql += " WHERE session = ?"
            args.append(session)
        with self._db() as con:
            rows = con.execute(sql + " ORDER BY id", args).fetchall()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(LOG_COLUMNS)
        writer.writerows([tuple(r) for r in rows])
        return buf.getvalue()

    def clear(self, session=None):
        with self._db() as con:
            if session:
                con.execute("DELETE FROM vehicle_log WHERE session = ?", (session,))
            else:
                con.execute("DELETE FROM vehicle_log")


# ---------------------------------------------------------------------------
# Model performance numbers for the dashboards
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
    """Detector, OCR and system metrics: from the local evaluation files when present,
    otherwise from EVALUATION_DEFAULTS (the values measured during the project)."""
    d = EVALUATION_DEFAULTS
    out = {"detection": dict(d["detection"]), "ocr": {}, "system": {"test_images": d["test_images"]}}
    recall_025 = d["recall_at_025"]
    if DETECTOR_METRICS.exists():
        m = json.loads(DETECTOR_METRICS.read_text())
        tm = m["test_metrics"]
        out["detection"] = {"mAP50": tm["mAP50"], "mAP50_95": tm["mAP50-95"],
                            "precision": tm["precision"], "recall": tm["recall"]}
        out["system"]["test_images"] = m["dataset"]["test"]["images"]
    if REPORT_NUMBERS.exists():
        for s in json.loads(REPORT_NUMBERS.read_text()).get("threshold_sweep", []):
            if abs(s["conf"] - 0.25) < 1e-6:
                recall_025 = s["recall"]

    current = {}
    out["ocr"]["valid_format_rate"] = d["valid_format_rate"]
    if OCR_RESULTS.exists():
        with open(OCR_RESULTS, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        current = {r["crop_file"]: r["text"] for r in rows}
        texts = [r["text"] for r in rows if r["text"]]
        if texts:
            out["ocr"]["valid_format_rate"] = sum(check_format(t)[0] for t in texts) / len(texts)

    sample = d["ocr_dev_sample"]
    source = f"{sample['plates']}-plate development sample"
    exact, chars = sample["exact"] / sample["plates"], sample["char_accuracy"]
    if MANUAL_CHECK.exists():
        with open(MANUAL_CHECK, newline="", encoding="utf-8") as f:
            filled = [r for r in csv.DictReader(f) if r["correct_text"].strip() not in ("", "-")]
        if filled:
            ex = errs = tot = 0
            for r in filled:
                ocr = "".join(c for c in current.get(r["crop_file"], r["ocr_text"]).upper() if c.isalnum())
                truth = "".join(c for c in r["correct_text"].upper() if c.isalnum())
                dist = edit_distance(ocr, truth)
                ex += dist == 0
                errs += dist
                tot += len(truth)
            source, exact, chars = f"manual check ({len(filled)} plates)", ex / len(filled), 1 - errs / max(1, tot)
    out["ocr"].update({"plate_accuracy": exact, "char_accuracy": chars, "cer": 1 - chars, "source": source})
    out["system"]["end_to_end"] = recall_025 * exact
    sp = d["pipeline_speed"]
    out["system"].update({"latency_ms": sp["ms_per_image"], "fps": sp["fps"], "speed_device": sp["device"]})
    return out
