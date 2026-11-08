"""
PlateScan - ALPR dashboard backend (FastAPI)
============================================
Serves the HTML dashboard (static/index.html) and runs the recognition
pipeline from pipeline.py.

Run from the repository folder:
    python platescan/server.py
then open http://127.0.0.1:8000 in your browser.

Everything runs locally; no image leaves this computer.
(For the Streamlit version of the dashboard, see streamlit_app.py.)
"""

import base64
import io
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402  (shared recognition pipeline)

STATIC_DIR = pl.APP_DIR / "static"
LOG = pl.VehicleLog(pl.APP_DIR / "platescan.db")
SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
MODELS = None


def to_b64(img, ext=".jpg", quality=85) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    mime = "jpeg" if ext == ".jpg" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf).decode() if ok else ""


@asynccontextmanager
async def lifespan(_app):
    """Load the models once when the server starts."""
    global MODELS
    print("Loading models...")
    MODELS = pl.Models()
    print(f"Models ready on {MODELS.device_name}. Open http://127.0.0.1:8000")
    yield


app = FastAPI(title="PlateScan", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": MODELS.device_name if MODELS else None,
            "models": {"plate": pl.PLATE_WEIGHTS.name, "vehicle": pl.VEHICLE_WEIGHTS.name, "ocr": pl.OCR_MODEL},
            "plate_conf": pl.PLATE_CONF, "review_ocr_conf": pl.REVIEW_OCR_CONF}


@app.get("/metrics")
def metrics():
    return pl.performance_numbers()


@app.post("/predict")
def predict(image: UploadFile = File(...), source: str = Form("upload")):
    try:
        img = pl.load_image(image.file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    t_start = time.perf_counter()
    plates, vehicles, timings, vis = pl.recognize(MODELS, img)
    total = (time.perf_counter() - t_start) * 1000

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOG.add(plates, ts, source, SESSION_ID)

    # numpy views -> base64 PNGs for the browser
    for p in plates:
        p["preprocess"] = [to_b64(v, ".png") for v in p.pop("views")]

    best = plates[0] if plates else None
    scale = 1280 / max(vis.shape[:2])
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
        "annotated": to_b64(vis if scale >= 1 else cv2.resize(vis, None, fx=scale, fy=scale)),
    }


@app.get("/log")
def get_log(search: str = "", limit: int = 200):
    return {"rows": LOG.rows(search, limit=limit), "session": LOG.stats(SESSION_ID)}


@app.get("/log.csv")
def export_csv():
    return StreamingResponse(iter([LOG.to_csv()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=platescan_log.csv"})


@app.delete("/log")
def clear_log():
    LOG.clear()
    return {"cleared": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
