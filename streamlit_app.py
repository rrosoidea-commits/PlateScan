"""
PlateScan - Streamlit app
=========================
Shows the original PlateScan dashboard page (platescan/static/index.html) as a
full-page Streamlit component, so the deployed app looks exactly like the
FastAPI version. The page sends its requests (health, metrics, predict, log,
clear, csv) to this script, which answers them with the shared pipeline
(platescan/pipeline.py).

Run locally (from the repository folder):
    streamlit run streamlit_app.py

Streamlit Community Cloud:
    Main file path: streamlit_app.py   (Python 3.12 in Advanced settings)
"""

import base64
import sys
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / "platescan"))
import pipeline as pl  # noqa: E402


def logo_icon(size=64):
    """The PlateScan logo (amber rounded square with a plate) as a PIL image, for the browser tab."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # amber tile: 44 px with 10 px corners on the page
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=round(size * 10 / 44), fill=(242, 182, 50, 255))
    # plate icon: a 24-unit SVG drawn at 24 px inside the 44 px tile (same proportions here)
    k = size / 44                     # pixels per page pixel
    off = (size - 24 * k) / 2         # centers the 24-unit icon
    p = lambda u: off + u * k         # icon units -> pixels
    ink, w = (42, 31, 5, 255), max(2, round(2 * k))
    d.rounded_rectangle([p(2.5), p(6), p(21.5), p(18)], radius=2.5 * k, outline=ink, width=w)
    d.line([p(6.5), p(12), p(9.5), p(12)], fill=ink, width=w)
    d.line([p(11.5), p(12), p(17.5), p(12)], fill=ink, width=w)
    return img


st.set_page_config(page_title="PlateScan", page_icon=logo_icon(), layout="wide",
                   initial_sidebar_state="collapsed")

# Hide Streamlit's own header, menu and padding so the dashboard fills the page
st.markdown("""
<style>
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"], .stApp { background: #0e1218; }
iframe { display: block; border: 0; }
</style>
""", unsafe_allow_html=True)

# The dashboard page, served by Streamlit as a custom component
dashboard = components.declare_component("platescan_dashboard", path=str(pl.APP_DIR / "static"))


@st.cache_resource(show_spinner="Loading PlateScan models (first start can take a minute)...")
def get_models():
    return pl.Models()


@st.cache_resource
def get_log():
    return pl.VehicleLog(pl.APP_DIR / "platescan_streamlit.db")


models = get_models()
log = get_log()

# Each browser session sees only its own log rows (important on a shared cloud app)
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.response = None
    st.session_state.last_request = None
SESSION = st.session_state.session_id


def handle(method: str, params: dict):
    """Answer one request from the dashboard page (same data as the FastAPI endpoints)."""
    if method == "health":
        return pl.health_payload(models)
    if method == "metrics":
        return pl.performance_numbers()
    if method == "predict":
        data_url = params.get("image", "")
        data = base64.b64decode(data_url.split(",", 1)[1] if "," in data_url else data_url)
        return pl.predict_payload(models, data, params.get("source", "upload"), log, SESSION)
    if method == "log":
        return {"rows": log.rows(params.get("search", ""), session=SESSION), "session": log.stats(SESSION)}
    if method == "clear":
        log.clear(SESSION)
        return {"cleared": True}
    if method == "csv":
        return {"csv": log.to_csv(SESSION)}
    raise ValueError(f"Unknown request: {method}")


request = dashboard(response=st.session_state.response, key="platescan", default=None)

# Handle each new request once, then re-render the page with the answer
if request and request.get("id") != st.session_state.last_request:
    st.session_state.last_request = request["id"]
    try:
        st.session_state.response = {"id": request["id"],
                                     "data": handle(request.get("method", ""), request.get("params") or {})}
    except Exception as e:  # shown in the page's error line
        st.session_state.response = {"id": request["id"], "error": str(e)}
    st.rerun()
