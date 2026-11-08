"""
PlateScan - Streamlit dashboard
===============================
Streamlit version of the PlateScan ALPR dashboard (same pipeline as
platescan/server.py, shared through platescan/pipeline.py).

Run locally (from the repository folder):
    streamlit run streamlit_app.py

Streamlit Community Cloud:
    Main file path: streamlit_app.py   (Python 3.12 in Advanced settings)
"""

import hashlib
import html
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "platescan"))
import pipeline as pl  # noqa: E402

st.set_page_config(page_title="PlateScan", page_icon="🚘", layout="wide")

# ---------------------------------------------------------------------------
# Styles for the plate card and small labels (the rest uses Streamlit's theme)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1250px; }
.plate { background:#f4f4ec; color:#15181d; border:3px solid #d9dbd2; border-radius:8px;
         padding:10px 12px 14px; text-align:center; margin:4px auto 14px; max-width:270px; }
.plate .tag { color:#1f7a3a; font-size:11px; font-weight:700; letter-spacing:.28em; }
.plate .num { font:700 32px/1.25 ui-monospace,Consolas,monospace; letter-spacing:.08em; margin-top:4px; }
.badge { display:inline-block; font-size:12.5px; padding:2px 10px; border-radius:999px; border:1px solid; }
.badge.ok { color:#34c78a; border-color:#34c78a; } .badge.review { color:#f0a35a; border-color:#f0a35a; }
.badge.none { color:#f06b6b; border-color:#f06b6b; } .badge.wait { color:#9aa3ae; border-color:#9aa3ae; }
.lbl { font-size:12px; font-weight:700; letter-spacing:.12em; margin-bottom:.3rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Models and log (loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (first start can take a minute)...")
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
    st.session_state.processed = set()   # hashes of images already recognized
    st.session_state.result = None
SESSION = st.session_state.session_id


def run_recognition(data: bytes, source: str):
    """Recognize an image once (Streamlit reruns the script on every interaction)."""
    digest = hashlib.md5(data).hexdigest()
    if digest in st.session_state.processed:
        return
    try:
        img = pl.load_image(data)
    except ValueError as e:
        st.error(str(e))
        return
    t0 = time.perf_counter()
    plates, vehicles, timings, vis = pl.recognize(models, img)
    total = (time.perf_counter() - t0) * 1000
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.add(plates, ts, source, SESSION)
    st.session_state.processed.add(digest)
    st.session_state.result = {"plates": plates, "vehicles": vehicles, "timings": timings, "total_ms": total,
                               "annotated": vis, "timestamp": ts, "source": source,
                               "size": (img.shape[1], img.shape[0])}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
h1, h2 = st.columns([3, 2])
with h1:
    st.markdown("## 🚘 PlateScan")
    st.caption("AI-based license plate recognition and vehicle logging · GR1")
with h2:
    st.markdown(f"<div style='text-align:right; padding-top:1.2rem'>"
                f"<span class='badge ok'>● Model connected · {html.escape(models.device_name)}</span>&nbsp;"
                f"<span class='badge wait'>YOLO26 · fast-plate-ocr · SQLite</span></div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input source + plate reading
# ---------------------------------------------------------------------------
col_in, col_read = st.columns([1.65, 1], gap="large")

with col_in:
    with st.container(border=True):
        st.subheader("Input source")
        tab_up, tab_cam = st.tabs(["Upload image", "Camera snapshot"])
        with tab_up:
            up = st.file_uploader("Upload a photo of a vehicle or plate",
                                  type=["jpg", "jpeg", "png", "bmp", "webp"])
            if up is not None:
                run_recognition(up.getvalue(), "upload")
        with tab_cam:
            st.caption("Allow camera access, point it at a plate (hold it close), then take a photo.")
            cam = st.camera_input("Take a photo")
            if cam is not None:
                run_recognition(cam.getvalue(), "camera")

        res = st.session_state.result
        if res:
            st.image(res["annotated"], channels="BGR", width="stretch",
                     caption=f"Result · {res['source']} · {res['size'][0]}×{res['size'][1]} px · "
                             "blue = vehicle, amber = plate")
        else:
            st.info("Upload an image or take a camera snapshot to run plate recognition.")

res = st.session_state.result
best = res["plates"][0] if res and res["plates"] else None

with col_read:
    with st.container(border=True):
        if not res:
            badge = "<span class='badge wait'>Waiting</span>"
        elif not best:
            badge = "<span class='badge none'>No plate found</span>"
        elif best["status"] == "OK":
            badge = "<span class='badge ok'>Logged</span>"
        else:
            badge = f"<span class='badge review'>Review: {html.escape(', '.join(best['review_reasons']))}</span>"
        st.markdown(f"#### Plate reading &nbsp; {badge}", unsafe_allow_html=True)

        if best:
            tag = "PHILIPPINES" if best["format_valid"] else (best["region"] or "Unknown region").upper()
            num = best["plate"] or "?"
        elif res:
            tag, num = "NO PLATE DETECTED", "--- ----"
        else:
            tag, num = "PHILIPPINES", "--- ----"
        st.markdown(f"<div class='plate'><div class='tag'>{html.escape(tag)}</div>"
                    f"<div class='num'>{html.escape(num)}</div></div>", unsafe_allow_html=True)

        ocr_c = best["ocr_conf"] if best else 0.0
        det_c = best["plate_conf"] if best else 0.0
        st.progress(ocr_c, text=f"OCR confidence: {ocr_c:.2f}" if best else "OCR confidence: —")
        st.progress(det_c, text=f"Plate detection confidence: {det_c:.2f}" if best else "Plate detection confidence: —")

        m1, m2 = st.columns(2)
        if best:
            veh = best["vehicle"] or (res["vehicles"][0]["label"] if res["vehicles"] else "—")
            m1.metric("Vehicle", veh, f"conf {best['vehicle_conf']:.2f}" if best["vehicle_conf"] else None,
                      delta_color="off")
            m2.metric("Format check", "✓ Valid PH" if best["format_valid"] else "✗ Not PH",
                      best["format"], delta_color="off")
        else:
            m1.metric("Vehicle", "—")
            m2.metric("Format check", "—")
        m3, m4 = st.columns(2)
        if res:
            m3.metric("Timestamp", res["timestamp"].split(" ")[1], res["source"], delta_color="off")
            m4.metric("Speed", f"{res['total_ms']:.0f} ms", f"{1000 / res['total_ms']:.0f} FPS equivalent",
                      delta_color="off")
        else:
            m3.metric("Timestamp", "—")
            m4.metric("Speed", "—")

        st.markdown("**Pipeline**")
        t = res["timings"] if res else {}
        n_pl = len(res["plates"]) if res else 0
        steps = [
            ("1 · Vehicle detection", f"{t['vehicle']:.0f} ms · {len(res['vehicles'])} found" if res else "–"),
            ("2 · Plate localization", f"{t['plate']:.0f} ms · {n_pl} found" if res else "–"),
            ("3 · Preprocessing", (f"{t['preprocess']:.0f} ms" if n_pl else "skipped") if res else "–"),
            ("4 · OCR", (f"{t['ocr']:.0f} ms" if n_pl else "skipped") if res else "–"),
            ("5 · Format check + log", (f"{n_pl} row{'s' if n_pl != 1 else ''}" if n_pl else "nothing") if res else "–"),
        ]
        st.markdown("\n".join(f"- {'✅' if res and v not in ('skipped', 'nothing') else '⬜'} {name} — `{v}`"
                              for name, v in steps))
        if res and len(res["plates"]) > 1:
            st.caption("Other plates in this image: " + ", ".join(p["plate"] or "?" for p in res["plates"][1:]))

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("Image preprocessing")
    st.caption("OpenCV views of the main plate · the OCR model reads the color crop (step 1)")
    captions = [("1 · Plate crop", "Cut from the full-resolution image"),
                ("2 · Grayscale", "Drops color, keeps character shape"),
                ("3 · Contrast (CLAHE)", "Lifts faded or shadowed characters"),
                ("4 · Threshold", "Black-and-white (Otsu)")]
    cols = st.columns(4)
    for i, (c, (title, sub)) in enumerate(zip(cols, captions)):
        with c:
            if best:
                st.image(best["views"][i], channels="BGR" if i == 0 else "GRAY", width="stretch")
            else:
                st.markdown("<div style='height:70px;border:1px dashed #555;border-radius:8px;"
                            "display:grid;place-items:center;color:#888'>—</div>", unsafe_allow_html=True)
            st.markdown(f"**{title}**  \n<small>{sub}</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
perf = pl.performance_numbers()
with st.container(border=True):
    st.subheader("Model performance (test split)")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("<div class='lbl' style='color:#5aa2f5'>PLATE DETECTION</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("mAP@0.5", f"{perf['detection']['mAP50']:.3f}")
        b.metric("mAP@0.5:0.95", f"{perf['detection']['mAP50_95']:.3f}")
        a.metric("Precision", f"{perf['detection']['precision']:.3f}")
        b.metric("Recall", f"{perf['detection']['recall']:.3f}")
        st.caption("YOLO26n · 593 test images · 743 plates")
    with p2:
        st.markdown("<div class='lbl' style='color:#f2b632'>PLATE READING (OCR)</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Plate accuracy", f"{100 * perf['ocr']['plate_accuracy']:.0f}%")
        b.metric("Character error rate", f"{100 * perf['ocr']['cer']:.1f}%")
        a.metric("Character accuracy", f"{100 * perf['ocr']['char_accuracy']:.1f}%")
        b.metric("Valid PH-format rate", f"{100 * perf['ocr']['valid_format_rate']:.0f}%")
        st.caption(f"fast-plate-ocr · accuracy from {perf['ocr']['source']}")
    with p3:
        st.markdown("<div class='lbl' style='color:#34c78a'>SYSTEM</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("End-to-end accuracy", f"{100 * perf['system']['end_to_end']:.0f}%")
        b.metric("Avg latency / image", f"{perf['system']['latency_ms']:.1f} ms")
        a.metric("Throughput", f"{perf['system']['fps']:.0f} FPS")
        b.metric("Test images", perf["system"]["test_images"])
        st.caption(f"Speed measured on {perf['system']['speed_device']} · this app runs on {models.device_name}")

    stats = log.stats(SESSION)
    s1, s2, s3 = st.columns(3)
    s1.metric("Vehicles logged (this session)", stats["logged"])
    s2.metric("Avg OCR confidence (this session)",
              "—" if stats["avg_ocr_conf"] is None else f"{stats['avg_ocr_conf']:.2f}")
    s3.metric("Flagged for review", stats["flagged"])

# ---------------------------------------------------------------------------
# Vehicle log
# ---------------------------------------------------------------------------
with st.container(border=True):
    lc1, lc2, lc3, lc4 = st.columns([1.3, 1.4, 0.8, 0.8], vertical_alignment="bottom")
    lc1.subheader("Vehicle log")
    search = lc2.text_input("Search plate", placeholder="e.g. NBC")
    lc3.download_button("Export CSV", log.to_csv(SESSION), file_name="platescan_log.csv",
                        mime="text/csv", width="stretch")
    if lc4.button("Clear", width="stretch"):
        log.clear(SESSION)
        st.session_state.processed = set()
        st.session_state.result = None
        st.rerun()

    rows = log.rows(search, session=SESSION)
    if rows:
        st.dataframe(
            [{"#": r["id"], "Timestamp": r["timestamp"], "Plate": r["plate"] or "?", "Vehicle": r["vehicle"] or "—",
              "OCR conf.": r["ocr_conf"], "Det. conf.": r["det_conf"], "Source": r["source"],
              "Status": ("✓ OK" if r["status"] == "OK" else "⚠ Review") + (f" ({r['reasons']})" if r["reasons"] else "")}
             for r in rows],
            hide_index=True, width="stretch",
            column_config={"OCR conf.": st.column_config.NumberColumn(format="%.2f"),
                           "Det. conf.": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.caption("No plates match this search." if search else
                   "No vehicles logged yet. Upload an image or take a snapshot to add the first row.")
    st.caption(f"Each row = one recognized plate, stored in SQLite. Rows are flagged for review when OCR confidence "
               f"is below {pl.REVIEW_OCR_CONF:.2f}, no text is read, or the text is not a Philippine plate format. "
               "Each browser session sees only its own rows; on Streamlit Cloud the log resets when the app restarts.")

# ---------------------------------------------------------------------------
# How it works + last result
# ---------------------------------------------------------------------------
b1, b2 = st.columns(2)
with b1:
    with st.expander("How it works", expanded=False):
        st.markdown(
            "1. **Vehicle detection** — YOLO26n pretrained on COCO (car, motorcycle, bus, truck)\n"
            "2. **Plate localization** — YOLO26n fine-tuned on ~16,900 plate images (`models/…best.pt`)\n"
            "3. **Preprocessing** — crop with a 5% margin (grayscale / CLAHE / threshold shown for inspection)\n"
            "4. **OCR** — fast-plate-ocr `cct-xs-v2-global-model` → regex check for Philippine formats\n"
            "5. **Log** — plate, timestamp and confidences saved to SQLite\n\n"
            "Philippine formats checked: `ABC 1234`, `ABC 123`, `123 ABC` (verify against current LTO rules).")
with b2:
    with st.expander("Last result (JSON)", expanded=False):
        if res:
            st.json({"plate": best["plate"] if best else None,
                     "ocr_conf": best["ocr_conf"] if best else None,
                     "plate_conf": best["plate_conf"] if best else None,
                     "vehicle": best["vehicle"] if best else None,
                     "vehicle_conf": best["vehicle_conf"] if best else None,
                     "plate_box": best["plate_box"] if best else None,
                     "timings_ms": {k: round(v, 1) for k, v in res["timings"].items()},
                     "total_ms": round(res["total_ms"], 1),
                     "plates": [{k: v for k, v in p.items() if k != "views"} for p in res["plates"]]})
        else:
            st.caption("No image processed yet.")

st.caption("PlateScan · Computer Vision professional elective · GR1")
