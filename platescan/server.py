"""
PlateScan - ALPR dashboard backend (FastAPI)
============================================
Serves the dashboard page (static/index.html) and runs the recognition
pipeline from pipeline.py.

Run from the repository folder:
    python platescan/server.py
then open http://127.0.0.1:8000 in your browser.

Everything runs locally; no image leaves this computer.
The same page is used by the Streamlit app (streamlit_app.py).
"""

import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402  (shared recognition pipeline)

STATIC_DIR = pl.APP_DIR / "static"
LOG = pl.VehicleLog(pl.APP_DIR / "platescan.db")
SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
MODELS = None


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
    return pl.health_payload(MODELS)


@app.get("/metrics")
def metrics():
    return pl.performance_numbers()


@app.post("/predict")
def predict(image: UploadFile = File(...), source: str = Form("upload")):
    try:
        return pl.predict_payload(MODELS, image.file.read(), source, LOG, SESSION_ID)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
