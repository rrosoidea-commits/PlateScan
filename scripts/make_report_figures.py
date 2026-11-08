"""Build report figures + extra analyses for RESULTS_DOCUMENTATION.md.

Reads only existing result files (and runs the trained detector on the test split once
for the per-size analysis). Writes PNGs to plate_detection/report_figures and a JSON of
computed numbers next to this script.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the repository folder
sys.path.insert(0, str(ROOT))  # so AI_PlateNumber can be imported

# ---- palette (dataviz reference instance, light mode) ----
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
GOOD, SERIOUS, CRITICAL = "#0ca30c", "#ec835a", "#d03b3b"

BASE = ROOT / "plate_detection"
RUN = BASE / "runs" / "yolo26n_640_e50"
FIG = BASE / "report_figures"
OUT_JSON = FIG / "report_numbers.json"


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK2,
        "lines.linewidth": 2,
    })
    return plt


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    print("saved", name)


def bar_rounded(ax, x, h, width, color, horizontal=False, left=0):
    """Thin bars with 4px-ish rounded data end, anchored to baseline."""
    from matplotlib.patches import FancyBboxPatch
    if horizontal:
        bars = ax.barh(x, h, height=width, left=left, color=color, edgecolor=SURF, linewidth=2)
    else:
        bars = ax.bar(x, h, width=width, color=color, edgecolor=SURF, linewidth=2)
    return bars


def main():
    import numpy as np
    plt = style()
    FIG.mkdir(parents=True, exist_ok=True)
    numbers = {}

    # ------------------------------------------------------------------ results.csv
    with open(RUN / "results.csv", newline="") as f:
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    ep = [int(r["epoch"]) for r in rows]
    col = lambda k: [r[k] for r in rows]
    best_i = max(range(len(rows)), key=lambda i: rows[i]["metrics/mAP50-95(B)"])
    best_ep = ep[best_i]

    # Fig: losses (small multiples: box loss, cls loss, l1 loss) train vs val
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ["box_loss", "cls_loss", "l1_loss"],
                              ["Box loss", "Classification loss", "L1 (coordinate) loss"]):
        ax.plot(ep, col(f"train/{key}"), color=S1, label="Train")
        ax.plot(ep, col(f"val/{key}"), color=S2, label="Validation")
        ax.axvline(40.5, color=MUTED, ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        tr_y, va_y = col(f"train/{key}")[-1], col(f"val/{key}")[-1]
        lo, hi = ax.get_ylim()
        min_gap = 0.07 * (hi - lo)
        if abs(va_y - tr_y) < min_gap:  # keep the two end labels apart
            mid = (va_y + tr_y) / 2
            va_y, tr_y = (mid + min_gap / 2, mid - min_gap / 2) if va_y >= tr_y else (mid - min_gap / 2, mid + min_gap / 2)
        ax.text(ep[-1] + 0.6, tr_y, "Train", color=INK2, va="center", fontsize=10)
        ax.text(ep[-1] + 0.6, va_y, "Val", color=INK2, va="center", fontsize=10)
        ax.set_xlim(0, 55)
    axes[0].text(41.2, axes[0].get_ylim()[1] * 0.97, "mosaic off", color=MUTED, fontsize=9, va="top")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
    fig.tight_layout()
    save(fig, "fig_training_losses.png")
    plt.close(fig)

    # Fig: metrics (small multiples: P & R ; mAP50 & mAP50-95)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.plot(ep, col("metrics/precision(B)"), color=S1, label="Precision")
    ax.plot(ep, col("metrics/recall(B)"), color=S2, label="Recall")
    ax.set_title("Validation precision & recall"); ax.set_xlabel("Epoch"); ax.set_ylim(0.88, 1.0)
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(ep, col("metrics/mAP50(B)"), color=S1, label="mAP@0.5")
    ax.plot(ep, col("metrics/mAP50-95(B)"), color=S2, label="mAP@0.5:0.95")
    ax.scatter([best_ep], [rows[best_i]["metrics/mAP50-95(B)"]], s=60, color=S2, edgecolor=SURF, linewidth=2, zorder=5)
    ax.annotate(f"best.pt (epoch {best_ep})\n{rows[best_i]['metrics/mAP50-95(B)']:.3f}",
                (best_ep, rows[best_i]["metrics/mAP50-95(B)"]), xytext=(30, 0.72),
                color=INK2, fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_title("Validation mAP"); ax.set_xlabel("Epoch"); ax.set_ylim(0.65, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.6))
    for a in axes:
        a.axvline(40.5, color=MUTED, ls="--", lw=1)
    fig.tight_layout()
    save(fig, "fig_training_metrics.png")
    plt.close(fig)

    numbers["training"] = {
        "best_epoch": best_ep,
        "first": {k: rows[0][k] for k in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]},
        "milestones": {e: rows[e - 1]["metrics/mAP50-95(B)"] for e in [1, 5, 10, 20, 30, 40, 50]},
        "val_box_min": min(col("val/box_loss")), "val_box_last": col("val/box_loss")[-1],
        "train_box_ep40": rows[39]["train/box_loss"], "train_box_ep41": rows[40]["train/box_loss"],
        "hours": rows[-1]["time"] / 3600,
    }

    # ------------------------------------------------------------------ val vs test
    metrics = json.loads((BASE / "outputs" / "yolo26n_640_e50_metrics.json").read_text())
    t = metrics["training"]; tm = metrics["test_metrics"]
    val_f1 = 2 * t["val_precision"] * t["val_recall"] / (t["val_precision"] + t["val_recall"])
    names = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    val_v = [t["val_precision"], t["val_recall"], val_f1, t["val_mAP50"], t["val_mAP50-95"]]
    test_v = [tm["precision"], tm["recall"], tm["f1"], tm["mAP50"], tm["mAP50-95"]]
    # Dot (dumbbell) plot: a non-zero axis is honest for dots, unlike for bars
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(names))[::-1]
    for yi, v, t_ in zip(y, val_v, test_v):
        ax.plot([v, t_], [yi, yi], color=AXIS, lw=2, zorder=1)
    ax.scatter(val_v, y, s=90, color=S1, edgecolor=SURF, linewidth=2, zorder=3, label="Validation (1,155 images)")
    ax.scatter(test_v, y, s=90, color=S2, edgecolor=SURF, linewidth=2, zorder=3, label="Test (593 images)")
    for yi, v, t_ in zip(y, val_v, test_v):
        lo_v, hi_v = min(v, t_), max(v, t_)
        ax.text(lo_v - 0.004, yi, f"{lo_v:.3f}", ha="right", va="center", fontsize=9, color=INK2)
        ax.text(hi_v + 0.004, yi, f"{hi_v:.3f}", ha="left", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, names); ax.set_xlim(0.76, 1.0); ax.grid(axis="y", visible=False)
    ax.set_title("Detector: validation (best epoch) vs. test split")
    from matplotlib.patches import Patch
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, -0.32))
    fig.tight_layout()
    save(fig, "fig_val_vs_test.png")
    plt.close(fig)
    numbers["val_f1"] = val_f1

    # ------------------------------------------------------------------ test-set analysis with the detector
    import cv2
    import AI_PlateNumber as det
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "yolo26n_640_e50_best.pt"))
    test_img = det.DATA_DIR / "test" / "images"
    test_lbl = det.DATA_DIR / "test" / "labels"
    gt_records = []     # (height_px_at_640, matched_bool)
    fp_confs, tp_confs = [], []
    all_preds = []      # per image: (preds, confs, gts) for threshold sweep
    for r in model.predict(source=str(test_img), imgsz=640, conf=0.01, device=0, stream=True, verbose=False):
        p = Path(r.path); h, w = r.orig_shape
        preds = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy()
        gts = [[(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
               for _, xc, yc, bw, bh in det.read_label_boxes(test_lbl / f"{p.stem}.txt")] \
            if (test_lbl / f"{p.stem}.txt").exists() else []
        all_preds.append((preds, confs, gts, 640 / max(h, w)))

    def match(preds, confs, gts, thr):
        keep = confs >= thr
        preds, confs = preds[keep], confs[keep]
        matched = {}
        fp = []
        tp = []
        for i in np.argsort(-confs):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                ix1, iy1 = max(preds[i][0], g[0]), max(preds[i][1], g[1])
                ix2, iy2 = min(preds[i][2], g[2]), min(preds[i][3], g[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                uni = (preds[i][2] - preds[i][0]) * (preds[i][3] - preds[i][1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
                iou = inter / uni if uni > 0 else 0
                if iou > best:
                    best, bj = iou, j
            if best >= 0.5:
                matched[bj] = confs[i]; tp.append(confs[i])
            else:
                fp.append(confs[i])
        return matched, tp, fp

    # threshold sweep
    sweep = []
    for thr in [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8]:
        TP = FP = FN = 0
        for preds, confs, gts, _ in all_preds:
            m, tp, fp = match(preds, confs, gts, thr)
            TP += len(tp); FP += len(fp); FN += len(gts) - len(m)
        P = TP / max(1, TP + FP); R = TP / max(1, TP + FN)
        sweep.append({"conf": thr, "TP": TP, "FP": FP, "FN": FN, "precision": P, "recall": R,
                      "f1": 2 * P * R / max(1e-9, P + R)})
    numbers["threshold_sweep"] = sweep

    # per-size recall at conf 0.25 + confidence distributions
    for preds, confs, gts, scale in all_preds:
        m, tp, fp = match(preds, confs, gts, 0.25)
        tp_confs += tp; fp_confs += fp
        for j, g in enumerate(gts):
            gt_records.append(((g[3] - g[1]) * scale, j in m))
    bins = [0, 20, 40, 80, 160, 10_000]
    labels = ["< 20 px", "20–40", "40–80", "80–160", "≥ 160 px"]
    size_rows = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        sel = [mm for hh, mm in gt_records if lo <= hh < hi]
        size_rows.append({"bucket": lab, "plates": len(sel), "found": sum(sel),
                          "recall": sum(sel) / len(sel) if sel else 0})
    numbers["recall_by_size"] = size_rows
    numbers["tp_conf_median"] = float(np.median(tp_confs)); numbers["fp_conf_median"] = float(np.median(fp_confs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    rec = [r["recall"] for r in size_rows]
    bar_rounded(ax, np.arange(len(labels)), rec, 0.55, S1)
    for i, r in enumerate(size_rows):
        ax.text(i, r["recall"] + 0.02, f"{100 * r['recall']:.0f}%\n({r['found']}/{r['plates']})",
                ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(labels)), labels); ax.set_ylim(0, 1.18); ax.grid(axis="x", visible=False)
    ax.set_yticks([0, .25, .5, .75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Plate height at 640 px"); ax.set_title("Test recall by plate size (conf ≥ 0.25)")
    ax = axes[1]
    b = np.linspace(0.25, 1.0, 31)
    ax.hist(tp_confs, bins=b, color=S1, alpha=0.9, label=f"Correct detections ({len(tp_confs)})", edgecolor=SURF, linewidth=1)
    ax.hist(fp_confs, bins=b, color=S2, alpha=0.95, label=f"False positives ({len(fp_confs)})", edgecolor=SURF, linewidth=1)
    ax.set_xlabel("Detector confidence"); ax.set_ylabel("Boxes"); ax.set_title("Confidence: correct vs. false detections")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_recall_by_size_and_confidence.png")
    plt.close(fig)

    # threshold sweep figure
    fig, ax = plt.subplots(figsize=(9, 4.2))
    th = [s["conf"] for s in sweep]
    ax.plot(th, [s["precision"] for s in sweep], color=S1, marker="o", ms=6, label="Precision")
    ax.plot(th, [s["recall"] for s in sweep], color=S2, marker="o", ms=6, label="Recall")
    ax.plot(th, [s["f1"] for s in sweep], color=S3, marker="o", ms=6, label="F1")
    ax.axvline(0.25, color=MUTED, ls="--", lw=1); ax.text(0.26, 0.42, "threshold used (0.25)", color=MUTED, fontsize=9)
    ax.set_xlabel("Confidence threshold"); ax.set_ylim(0.4, 1.02)
    ax.set_title("Precision / recall / F1 vs. confidence threshold (test split, IoU ≥ 0.5)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_threshold_sweep.png")
    plt.close(fig)

    # ------------------------------------------------------------------ dataset figure
    hs = {}
    for split, folder in det.SPLIT_FOLDERS.items():
        vals = []
        for lbl in (det.DATA_DIR / folder / "labels").glob("*.txt"):
            for _, _, _, _, hh in det.read_label_boxes(lbl):
                vals.append(hh * 640)  # all images are 640x640 (Roboflow stretch)
        hs[split] = np.array(vals)
    allh = np.concatenate(list(hs.values()))
    stats = metrics["dataset"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    splits = ["train", "val", "test"]
    imgs = [stats[s]["images"] for s in splits]
    bar_rounded(ax, np.arange(3), imgs, 0.5, S1)
    tot = sum(imgs)
    for i, v in enumerate(imgs):
        ax.text(i, v + 250, f"{v:,}\n({100 * v / tot:.1f}%)", ha="center", fontsize=10, color=INK2)
    ax.set_xticks(range(3), ["Train", "Valid", "Test"]); ax.set_ylim(0, 18500); ax.grid(axis="x", visible=False)
    ax.set_title("Images per split"); ax.set_ylabel("Images")
    ax = axes[1]
    ax.hist(np.clip(allh, 0, 640), bins=np.arange(0, 660, 20), color=S1, edgecolor=SURF, linewidth=1)
    ax.axvline(20, color=S2, lw=1.5, ls="--")
    ax.axvline(np.median(allh), color=MUTED, lw=1)
    top = ax.get_ylim()[1]
    ax.text(230, top * 0.92, f"- - 20 px line: {np.sum(allh < 20)} tiny plates ({100 * np.mean(allh < 20):.1f}%)",
            color=INK2, fontsize=9.5)
    ax.text(230, top * 0.84, f"—  median plate height: {np.median(allh):.0f} px", color=INK2, fontsize=9.5)
    ax.set_xlabel("Plate height in pixels (images are 640×640)"); ax.set_ylabel("Plates")
    ax.set_title(f"Plate size distribution (all splits, {len(allh):,} plates)")
    fig.tight_layout()
    save(fig, "fig_dataset.png")
    plt.close(fig)
    numbers["plate_height_median"] = float(np.median(allh))
    numbers["plate_height_pct"] = {s: {"lt20": float(np.mean(v < 20)), "ge160": float(np.mean(v >= 160))} for s, v in hs.items()}

    # ------------------------------------------------------------------ error breakdown
    ea = metrics["test_error_analysis"]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    vals = [ea["true_positives"], ea["missed_plates"], ea["false_positives"]]
    labs = ["Correct (TP)", "Missed (FN)", "False positive (FP)"]
    cols = [GOOD, CRITICAL, SERIOUS]
    for i, (v, l, c) in enumerate(zip(vals, labs, cols)):
        ax.barh(i, v, height=0.55, color=c, edgecolor=SURF, linewidth=2)
        ax.text(v + 8, i, f"{v}", va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(3), ["✔ " + labs[0], "✖ " + labs[1], "! " + labs[2]])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, 800)
    ax.set_title(f"Test detections at conf ≥ 0.25, IoU ≥ 0.5 ({ea['true_plates']} real plates)")
    fig.tight_layout()
    save(fig, "fig_error_breakdown.png")
    plt.close(fig)

    # ------------------------------------------------------------------ speed
    sp = metrics["speed"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    parts = [("Preprocess", sp["preprocess_ms"], S3), ("Inference", sp["inference_ms"], S1), ("Postprocess", sp["postprocess_ms"], S2)]
    left = 0
    for name, v, c in parts:
        ax.barh(0, v, left=left, height=0.5, color=c, edgecolor=SURF, linewidth=2, label=f"{name} {v:.1f} ms")
        left += v
    ax.text(left + 0.3, 0, f"{left:.1f} ms → {1000 / left:.0f} FPS", va="center", fontsize=10, color=INK2)
    ax.set_yticks([]); ax.set_xlim(0, 24); ax.set_xlabel("Milliseconds per image")
    ax.set_title("Detector time per image (batch 1, RTX 2050)"); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    pipes = [("EasyOCR", 17.6, 41.3), ("fast-plate-ocr", 20.8, 3.0)]
    for i, (n, d, o) in enumerate(pipes):
        ax.barh(i, d, height=0.5, color=S1, edgecolor=SURF, linewidth=2, label="Detection" if i == 0 else None)
        ax.barh(i, o, left=d, height=0.5, color=S2, edgecolor=SURF, linewidth=2, label="OCR" if i == 0 else None)
        ax.text(d + o + 0.8, i, f"{d + o:.1f} ms → {1000 / (d + o):.0f} FPS", va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(2), [p[0] for p in pipes]); ax.invert_yaxis(); ax.set_xlim(0, 80)
    ax.set_xlabel("Milliseconds per image"); ax.set_title("Full pipeline (detect + read)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_speed.png")
    plt.close(fig)

    # ------------------------------------------------------------------ OCR versions
    versions = ["EasyOCR v1\n(CLAHE, 96 px)", "EasyOCR v2\n(median, 160 px)", "fast-plate-ocr\n(cct-xs-v2)"]
    exact = [1 / 23, 7 / 23, 20 / 23]; chars = [0.558, 0.741, 0.980]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(3); w = 0.34
    bar_rounded(ax, x - w / 2, exact, w, S1); bar_rounded(ax, x + w / 2, chars, w, S2)
    for i in range(3):
        ax.text(x[i] - w / 2, exact[i] + 0.02, f"{round(exact[i] * 23)}/23\n{100 * exact[i]:.0f}%", ha="center", fontsize=9, color=INK2)
        ax.text(x[i] + w / 2, chars[i] + 0.02, f"{100 * chars[i]:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, versions); ax.set_ylim(0, 1.18); ax.grid(axis="x", visible=False)
    ax.set_yticks([0, .25, .5, .75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("OCR accuracy on 23 hand-labeled plate crops")
    ax.legend(handles=[Patch(color=S1, label="Plates read exactly"), Patch(color=S2, label="Characters correct")],
              loc="upper left")
    fig.tight_layout()
    save(fig, "fig_ocr_versions.png")
    plt.close(fig)

    # ------------------------------------------------------------------ OCR on whole test set
    def load(name):
        with open(BASE / "ocr_outputs" / name, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    fpo, easy = load("ocr_results.csv"), load("ocr_results_easyocr.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    lens = np.arange(0, 11)
    cf = [sum(1 for r in fpo if int(r["n_chars"]) == n) for n in lens]
    ce = [sum(1 for r in easy if int(r["n_chars"]) == n) for n in lens]
    w = 0.4
    ax.bar(lens - w / 2, ce, width=w, color=S1, edgecolor=SURF, linewidth=1.5, label="EasyOCR")
    ax.bar(lens + w / 2, cf, width=w, color=S2, edgecolor=SURF, linewidth=1.5, label="fast-plate-ocr")
    ax.axvspan(3.5, 8.5, color=GRID, alpha=0.5, zorder=0)
    ax.text(6, max(cf) * 1.02, "typical plate length (4–8)", ha="center", color=MUTED, fontsize=9)
    ax.set_xticks(lens); ax.set_xlabel("Characters read (0 = nothing read)"); ax.set_ylabel("Plates")
    ax.set_title(f"Length of text read ({len(fpo)} detected test plates)"); ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    b = np.linspace(0, 1, 21)
    ax.hist([float(r["ocr_conf"]) for r in easy if r["text"]], bins=b, color=S1, alpha=0.85, label="EasyOCR", edgecolor=SURF)
    ax.hist([float(r["ocr_conf"]) for r in fpo if r["text"]], bins=b, color=S2, alpha=0.85, label="fast-plate-ocr", edgecolor=SURF)
    ax.set_xlabel("OCR confidence"); ax.set_ylabel("Plates"); ax.set_title("OCR confidence distribution"); ax.legend(loc="upper center")
    fig.tight_layout()
    save(fig, "fig_ocr_testset.png")
    plt.close(fig)

    # OCR by plate size (fast-plate-ocr): share of 4-8 char reads and mean confidence
    def h640(r):
        return int(r["y2"]) - int(r["y1"])  # test images are 640x640
    ocr_size = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        sel = [r for r in fpo if lo <= h640(r) / 1.1 < hi]  # crops include a 5% margin each side
        ok = [r for r in sel if 4 <= int(r["n_chars"]) <= 8]
        ocr_size.append({"bucket": lab, "plates": len(sel), "plausible": len(ok),
                         "share": len(ok) / len(sel) if sel else 0,
                         "mean_conf": float(np.mean([float(r["ocr_conf"]) for r in sel])) if sel else 0})
    numbers["ocr_by_size"] = ocr_size
    numbers["ocr_counts"] = {
        "fpo": {"plates": len(fpo), "read": sum(1 for r in fpo if r["text"]),
                "plausible": sum(1 for r in fpo if 4 <= int(r["n_chars"]) <= 8),
                "conf_mean": float(np.mean([float(r["ocr_conf"]) for r in fpo if r["text"]])),
                "conf_ge_0_9": sum(1 for r in fpo if float(r["ocr_conf"]) >= 0.9)},
        "easy": {"plates": len(easy), "read": sum(1 for r in easy if r["text"]),
                 "plausible": sum(1 for r in easy if 4 <= int(r["n_chars"]) <= 8),
                 "conf_mean": float(np.mean([float(r["ocr_conf"]) for r in easy if r["text"]])),
                 "conf_ge_0_9": sum(1 for r in easy if float(r["ocr_conf"]) >= 0.9)},
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    sh = [o["share"] for o in ocr_size]
    bar_rounded(ax, np.arange(len(labels)), sh, 0.55, S2)
    for i, o in enumerate(ocr_size):
        ax.text(i, o["share"] + 0.02, f"{100 * o['share']:.0f}%\n({o['plausible']}/{o['plates']})", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(labels)), labels); ax.set_ylim(0, 1.2); ax.grid(axis="x", visible=False)
    ax.set_yticks([0, .25, .5, .75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Detected plate height at 640 px")
    ax.set_title("fast-plate-ocr: share of plates read with a plausible length (4–8 chars)")
    fig.tight_layout()
    save(fig, "fig_ocr_by_size.png")
    plt.close(fig)

    OUT_JSON.write_text(json.dumps(numbers, indent=2, default=float))
    print(json.dumps(numbers, indent=2, default=float))


if __name__ == "__main__":
    main()
