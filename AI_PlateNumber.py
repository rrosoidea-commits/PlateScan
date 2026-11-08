"""
AI-based License Plate Recognition - Part 1: License Plate DETECTOR training
============================================================================
Trains an Ultralytics YOLO26 model to find license plates (bounding boxes).
OCR and tracking come later in a separate script.

Run from a terminal:
    python AI_PlateNumber.py

The script is split into numbered parts (functions). main() at the bottom
runs them in order.

IMPORTANT (Windows): the code that runs everything sits under
    if __name__ == "__main__":
Ultralytics loads images with several worker processes, and on Windows each
worker re-imports this file. Without that guard, every worker would start
the whole script again.
"""

import sys
import shutil
import subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # the repository folder

# Roboflow export (YOLO26 format) with train/, valid/, test/ and data.yaml.
# Download it from Roboflow and put it in data/ (see README.md).
DATA_DIR = ROOT / "data" / "License Plate Detection MERGED PROJECTS.v3-best-model.yolo26"
YAML_PATH = DATA_DIR / "data.yaml"

# Where this script saves training runs and the final files for the report
BASE_DIR = ROOT / "plate_detection"
RUNS_DIR = BASE_DIR / "runs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Charts are always saved as PNG files in OUTPUTS_DIR.
# SHOW_PLOTS = True also opens each chart in a window, and the script
# PAUSES until you close that window. Set it to False to run unattended.
SHOW_PLOTS = False

# Settings shared by several parts
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # image file types
SPLIT_FOLDERS = {"train": "train", "val": "valid", "test": "test"}  # split -> folder name
TRAIN_SIZE = 640  # image size used for training
TINY_PX = 20      # plates shorter than this (in px at 640) count as "tiny"

# ---------------------------------------------------------------------------
# Training settings (Part 5)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = "yolo26n.pt"  # pretrained on COCO; change to "yolo26s.pt" to compare
EPOCHS = 50       # at most 50 passes over the training set
PATIENCE = 15     # stop early if val mAP hasn't improved for 15 epochs
SEED = 42         # makes the run repeatable
# Background processes that load images. On Windows each one loads its own copy
# of PyTorch + CUDA (~1 GB of memory), and validation uses 2x this number.
# With 8 GB RAM, 4 workers crashed with "WinError 1455: paging file too small",
# so we use 1. If you have more free memory, try 2.
WORKERS = 1

# Each model gets its own run folder, e.g. runs/yolo26n_640_e50
RUN_NAME = f"{Path(MODEL_WEIGHTS).stem}_{TRAIN_SIZE}_e{EPOCHS}"
RUN_DIR = RUNS_DIR / RUN_NAME
BEST_PT = RUN_DIR / "weights" / "best.pt"
LAST_PT = RUN_DIR / "weights" / "last.pt"

# What Part 5 does when you run the script:
#   RETRAIN = False -> if best.pt already exists, skip training and reuse it
#   RETRAIN = True  -> train again from scratch (overwrites this run folder)
#   RESUME  = True  -> continue an interrupted run from last.pt
RETRAIN = False
RESUME = False


# ===========================================================================
# PART 1: Environment setup
# ---------------------------------------------------------------------------
# - Installs the CUDA (GPU) build of PyTorch if the PC has an NVIDIA GPU but
#   only the CPU build is installed. This is checked BEFORE importing torch,
#   because Windows can't replace PyTorch's files once they're loaded.
# - Installs ultralytics if missing (YOLO26 needs version 8.4.0 or newer).
# - Prints the Python / PyTorch / Ultralytics versions (for the report).
# - Picks the device (GPU or CPU) and a batch size that fits in memory.
#
# Why: a GPU trains often 10-50x faster than a CPU. Batch size is how many
# images are processed in one step. More GPU memory allows a bigger batch.
# A 4 GB GPU (e.g. RTX 2050) uses batch 8 to avoid "CUDA out of memory".
# ===========================================================================
def pip_install(*args):
    """Run 'pip install ...' using the same Python that runs this script."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def setup_environment():
    """Install what is missing, print versions, and return (device, device_name, batch)."""
    # --- 1) GPU build of PyTorch (Windows only) ---
    # nvidia-smi comes with the NVIDIA driver, so if it exists there is an NVIDIA GPU.
    has_nvidia_gpu = shutil.which("nvidia-smi") is not None
    try:
        torch_version = version("torch")  # read the version WITHOUT importing torch
    except PackageNotFoundError:
        torch_version = None

    on_windows = sys.platform == "win32"
    torch_is_cuda_build = torch_version is not None and "+cu" in torch_version

    if on_windows and has_nvidia_gpu and not torch_is_cuda_build:
        print(f"NVIDIA GPU found, but PyTorch is '{torch_version}' (no CUDA).")
        print("Installing the CUDA 12.8 build of PyTorch (about 2-3 GB download)...")
        # --force-reinstall replaces the CPU build; --no-deps skips reinstalling numpy etc.
        extra = ["--force-reinstall", "--no-deps"] if torch_version else []
        pip_install(*extra, "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")

    # --- 2) Ultralytics (installed AFTER torch so it keeps the CUDA build) ---
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics...")
        pip_install("-U", "ultralytics")
        import ultralytics

    import torch

    # --- Versions ---
    print("Python      :", sys.version.split()[0])
    print("PyTorch     :", torch.__version__)
    print("Ultralytics :", ultralytics.__version__)

    # YOLO26 models were added in ultralytics 8.4.0
    major, minor = (int(x) for x in ultralytics.__version__.split(".")[:2])
    if (major, minor) < (8, 4):
        print("WARNING: ultralytics is too old for YOLO26. Run: pip install -U ultralytics")

    # --- 3) Device and batch size ---
    if torch.cuda.is_available():
        device = 0  # first CUDA GPU
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # Rough rule for yolo26n at 640 px: more VRAM -> bigger batch
        if vram_gb >= 12:
            batch = 32
        elif vram_gb >= 6:
            batch = 16
        else:
            batch = 8
        print(f"CUDA GPU    : {device_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = "cpu"
        device_name = "CPU"
        batch = 8
        print("CUDA GPU    : not available, training will run on the CPU (very slow)")

    print(f"Using device={device}, batch={batch}")

    # Make sure the dataset is where we expect it
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"data.yaml not found at: {YAML_PATH}")
    print("Dataset     :", DATA_DIR)

    return device, device_name, batch


# ===========================================================================
# PART 2: Fix data.yaml
# ---------------------------------------------------------------------------
# Roboflow writes paths like "../train/images". Those are relative to the
# folder you LAUNCH training from, so they often break. We:
# - back up the original file once (data_original.yaml),
# - set "path:" to the dataset's ABSOLUTE folder,
# - make train/val/test RELATIVE to that folder,
# - print the class names and nc (number of classes) and check the folders.
# ===========================================================================
def fix_data_yaml():
    """Rewrite data.yaml with an absolute root; return the list of class names."""
    # Ultralytics has its own YAML reader/writer (no separate PyYAML needed)
    from ultralytics.utils import YAML

    # Output folders used by the later parts
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep one copy of the original file
    backup_path = DATA_DIR / "data_original.yaml"
    if not backup_path.exists():
        shutil.copy(YAML_PATH, backup_path)
        print("Backup saved:", backup_path.name)

    old_cfg = YAML.load(YAML_PATH)

    # New config: absolute root first, then paths relative to it,
    # then everything else from the old file (nc, names, roboflow info)
    data_cfg = {
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    for key, value in old_cfg.items():
        if key not in data_cfg:
            data_cfg[key] = value

    YAML.save(YAML_PATH, data_cfg)

    print("New data.yaml:")
    print(YAML_PATH.read_text())

    class_names = data_cfg["names"]
    num_classes = data_cfg["nc"]
    print("nc    :", num_classes)
    print("names :", class_names)

    # Quick checks: class count matches and every split folder exists
    assert num_classes == len(class_names), "nc does not match the number of names"
    for key in ["train", "val", "test"]:
        folder = DATA_DIR / data_cfg[key]
        assert folder.is_dir(), f"Missing folder: {folder}"
    print("All split folders found.")

    return class_names


# ---------------------------------------------------------------------------
# PART 2b: Fix label files that mix polygons and boxes
# ---------------------------------------------------------------------------
# This merged Roboflow dataset has two kinds of label rows:
#   box:     class xc yc w h              (5 values)
#   polygon: class x1 y1 x2 y2 x3 y3 ...  (more values)
# Files with ONLY boxes or ONLY polygons are fine (Ultralytics turns
# polygons into boxes itself). But ~4.7% of files MIX both kinds, and
# Ultralytics rejects those images as "corrupt" (it would silently drop
# ~800 images, including 35 test images).
# Fix: in those mixed files, replace each polygon row with its bounding box.
# A detector only uses boxes, so nothing is lost. The original files are
# copied to <split>/labels_original/ first.
# ---------------------------------------------------------------------------
def polygon_row_to_box_row(parts):
    """['0', x1, y1, x2, y2, ...] -> 'class xc yc w h' (bounding box of the polygon)."""
    cls = parts[0]
    coords = [float(v) for v in parts[1:]]
    xs, ys = coords[0::2], coords[1::2]
    x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))  # clip to the image
    y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
    xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = x_max - x_min, y_max - y_min
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def fix_mixed_labels():
    """Convert polygon rows to boxes in label files that mix both formats."""
    for split, folder in SPLIT_FOLDERS.items():
        lbl_dir = DATA_DIR / folder / "labels"
        backup_dir = DATA_DIR / folder / "labels_original"
        fixed = 0

        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            rows = [line.split() for line in lbl_path.read_text().splitlines() if line.strip()]
            is_box = [len(r) == 5 for r in rows]
            if not (any(is_box) and not all(is_box)):
                continue  # only boxes, only polygons, or empty -> already fine

            # Back up the original file once, then rewrite it with boxes only
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / lbl_path.name
            if not backup_path.exists():
                shutil.copy(lbl_path, backup_path)
            new_rows = [" ".join(r) if len(r) == 5 else polygon_row_to_box_row(r) for r in rows]
            lbl_path.write_text("\n".join(new_rows) + "\n")
            fixed += 1

        # Ultralytics caches the scanned labels; delete the cache so it re-reads them
        cache = DATA_DIR / folder / "labels.cache"
        if fixed and cache.exists():
            cache.unlink()
        print(f"{split:<6}: {fixed} mixed label files converted to boxes"
              + (f" (originals in {backup_dir.name}/)" if fixed else ""))


# ===========================================================================
# PART 3: Dataset sanity check
# ---------------------------------------------------------------------------
# Before spending hours training, check that the data is what we expect:
# - Images vs labels per split: every image should have a .txt label file
#   with the same name.
# - Missing or empty labels: YOLO treats both as "background" images (no
#   plates). A few are fine (they teach the model what is NOT a plate); a
#   lot would mean annotations were lost.
# - Box-size distribution: each plate's height is converted to pixels AFTER
#   resizing the image to 640 (longest side -> 640, like YOLO does). Plates
#   under 20 px tall are hard to detect, and too small for OCR later.
#
# Label format: one line per plate, "class x_center y_center width height",
# all normalized to 0-1. Polygon lines (class x1 y1 x2 y2 ...) are turned
# into their bounding box.
# ===========================================================================
def list_images(folder):
    """All image files in a folder, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def read_label_boxes(label_path):
    """Read a YOLO label file -> list of (class_id, x_center, y_center, w, h), normalized 0-1."""
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue  # skip blank lines
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:  # normal box
            xc, yc, w, h = coords
        else:                 # polygon -> bounding box
            xs, ys = coords[0::2], coords[1::2]
            xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append((cls, xc, yc, w, h))
    return boxes


def dataset_sanity_check():
    """Count images/labels/boxes per split and show plate sizes. Returns the counts per split."""
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    split_stats = {}  # counts per split (saved to the report in Part 10)
    box_heights = {}  # plate heights in px at 640, per split

    for split, folder in SPLIT_FOLDERS.items():
        img_dir = DATA_DIR / folder / "images"
        lbl_dir = DATA_DIR / folder / "labels"
        images = list_images(img_dir)
        label_files = list(lbl_dir.glob("*.txt"))

        missing, empty, heights = [], [], []
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                missing.append(img_path.name)
                continue
            boxes = read_label_boxes(lbl_path)
            if not boxes:
                empty.append(img_path.name)
                continue
            with Image.open(img_path) as im:  # only reads the file header, fast
                img_w, img_h = im.size
            scale = TRAIN_SIZE / max(img_w, img_h)  # longest side -> 640
            for _, _, _, w, h in boxes:
                heights.append(h * img_h * scale)

        box_heights[split] = np.array(heights)
        split_stats[split] = {
            "images": len(images),
            "label_files": len(label_files),
            "boxes": len(heights),
            "missing_label": len(missing),
            "empty_label": len(empty),
        }
        if missing:
            print(f"[{split}] {len(missing)} images WITHOUT a label file, e.g. {missing[:5]}")
        if empty:
            print(f"[{split}] {len(empty)} images with an EMPTY label (background), e.g. {empty[:5]}")

    # --- Counts table ---
    print(f"\n{'split':<6}{'images':>8}{'labels':>8}{'boxes':>8}{'missing':>9}{'empty':>7}")
    for split, s in split_stats.items():
        print(f"{split:<6}{s['images']:>8}{s['label_files']:>8}{s['boxes']:>8}"
              f"{s['missing_label']:>9}{s['empty_label']:>7}")
    total_images = sum(s["images"] for s in split_stats.values())
    for split, s in split_stats.items():
        print(f"{split}: {100 * s['images'] / total_images:.1f}% of {total_images} images")

    # --- Plate height distribution ---
    bins = [0, 10, 20, 40, 80, 160, np.inf]
    bin_names = ["<10", "10-20", "20-40", "40-80", "80-160", ">=160"]
    print(f"\nPlate height at {TRAIN_SIZE}px (number of boxes):")
    print(f"{'split':<6}" + "".join(f"{b:>9}" for b in bin_names) + f"{'tiny %':>9}")
    for split, h in box_heights.items():
        counts, _ = np.histogram(h, bins=bins)
        tiny_pct = 100 * np.mean(h < TINY_PX) if len(h) else 0
        print(f"{split:<6}" + "".join(f"{c:>9}" for c in counts) + f"{tiny_pct:>8.1f}%")

    all_heights = np.concatenate(list(box_heights.values()))
    print(f"\nMedian plate height: {np.median(all_heights):.1f}px, "
          f"tiny (<{TINY_PX}px): {np.sum(all_heights < TINY_PX)} of {len(all_heights)} boxes")

    # --- Histogram chart ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(np.clip(all_heights, 0, 200), bins=40, color="steelblue")
    ax.axvline(TINY_PX, color="red", linestyle="--", label=f"{TINY_PX}px (tiny)")
    ax.set_xlabel(f"Plate height in pixels at {TRAIN_SIZE} (values over 200 grouped at 200)")
    ax.set_ylabel("Number of boxes")
    ax.set_title("Plate size distribution (all splits)")
    ax.legend()
    chart_path = OUTPUTS_DIR / "box_height_distribution.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    print("Chart saved:", chart_path)
    if SHOW_PLOTS:
        plt.show()  # close the window to continue
    plt.close(fig)

    return split_stats


# ===========================================================================
# PART 4: Look at some training images
# ---------------------------------------------------------------------------
# Numbers alone don't show whether the boxes are CORRECT. We draw the
# ground-truth boxes on 6 random training images. Check that each box fits
# tightly around a plate and that no plate is left unlabeled.
# The fixed random seed (42) picks the same 6 images every run, which is
# handy for the report.
# ===========================================================================
def show_training_samples(class_names, n_samples=6, seed=42):
    """Draw the labeled boxes on random training images and save the grid."""
    import random
    import cv2
    import matplotlib.pyplot as plt

    random.seed(seed)
    train_img_dir = DATA_DIR / "train" / "images"
    train_lbl_dir = DATA_DIR / "train" / "labels"
    samples = random.sample(list_images(train_img_dir), n_samples)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, img_path in zip(axes.flat, samples):
        # OpenCV loads images as BGR; matplotlib expects RGB
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        thickness = max(2, img_w // 300)  # thicker lines on big images

        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        boxes = read_label_boxes(lbl_path) if lbl_path.exists() else []
        for cls, xc, yc, w, h in boxes:
            # normalized center/size -> pixel corners
            x1, y1 = int((xc - w / 2) * img_w), int((yc - h / 2) * img_h)
            x2, y2 = int((xc + w / 2) * img_w), int((yc + h / 2) * img_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(img, class_names[cls], (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ax.imshow(img)
        ax.set_title(f"{img_path.name[:35]}\n{len(boxes)} plate(s), {img_w}x{img_h}", fontsize=9)
        ax.axis("off")
        print(f"  {img_path.name}: {len(boxes)} plate(s), {img_w}x{img_h}")

    plt.tight_layout()
    chart_path = OUTPUTS_DIR / "train_samples.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    print("Image grid saved:", chart_path)
    if SHOW_PLOTS:
        plt.show()  # close the window to continue
    plt.close(fig)


# ===========================================================================
# PART 5: Train the detector
# ---------------------------------------------------------------------------
# Transfer learning: we start from YOLO26 weights pretrained on COCO, so the
# model already knows general shapes and edges and only has to learn what
# a license plate looks like. Augmentations are the Ultralytics defaults
# (mosaic, HSV color changes, horizontal flip, translate, scale).
#
# Expect a few hours on a laptop GPU with ~15k training images. Progress
# is printed every epoch. Outputs land in plate_detection/runs/<RUN_NAME>/:
#   weights/best.pt  - epoch with the best validation score (we use this)
#   weights/last.pt  - most recent epoch (used to resume)
#   results.csv / results.png - metrics per epoch (Part 6)
# ===========================================================================
def train_detector(device, batch):
    """Train YOLO26 on the plate dataset (or reuse/resume a previous run). Returns best.pt path."""
    from ultralytics import YOLO

    # --- Skip / resume logic so re-running the script doesn't retrain by accident ---
    if RESUME and LAST_PT.exists():
        print(f"Resuming interrupted training from {LAST_PT}")
        model = YOLO(str(LAST_PT))
        model.train(resume=True)  # reuses the original settings saved in last.pt
        return BEST_PT

    if BEST_PT.exists() and not RETRAIN:
        print(f"best.pt already exists, skipping training: {BEST_PT}")
        print("(Set RETRAIN = True at the top of the file to train again.)")
        return BEST_PT

    # --- Fresh training run ---
    print(f"Model     : {MODEL_WEIGHTS}")
    print(f"Run folder: {RUN_DIR}")
    print(f"Settings  : imgsz={TRAIN_SIZE}, epochs={EPOCHS}, patience={PATIENCE}, "
          f"batch={batch}, device={device}, seed={SEED}")

    model = YOLO(MODEL_WEIGHTS)  # downloads the pretrained weights the first time

    model.train(
        data=str(YAML_PATH),   # the data.yaml fixed in Part 2
        epochs=EPOCHS,
        imgsz=TRAIN_SIZE,
        batch=batch,
        device=device,
        patience=PATIENCE,
        seed=SEED,
        deterministic=True,    # same seed -> same results
        workers=WORKERS,
        project=str(RUNS_DIR), # absolute path -> runs/<RUN_NAME>
        name=RUN_NAME,
        exist_ok=True,         # reuse the folder instead of creating ..._2, ..._3
        plots=True,            # save results.png, label plots, etc.
    )

    print("\nTraining finished. Best weights:", BEST_PT)
    return BEST_PT


# ===========================================================================
# PART 6: Training curves
# ---------------------------------------------------------------------------
# results.png plots losses and metrics after every epoch. What to look for:
# - TRAIN losses (box, cls, l1) should go DOWN steadily.
#   (YOLO26 uses an "l1_loss" for box coordinates instead of the older
#   "dfl_loss", because it predicts boxes directly without NMS.)
# - VAL losses should go down too. If they flatten and then RISE while
#   the train loss keeps falling, the model is OVERFITTING (memorizing
#   the training images instead of learning plates in general).
# - precision / recall / mAP50 / mAP50-95 should go UP and level off.
#   A plateau means more epochs would not help much.
# - A sudden drop in the train losses 10 epochs before the end is NORMAL:
#   that's "close_mosaic", where mosaic augmentation is switched off and
#   the training images suddenly become easier.
# - best.pt comes from the epoch with the best validation score, which
#   is not always the last epoch.
# ===========================================================================
def show_training_curves():
    """Show results.png and summarize results.csv. Returns a dict for the report."""
    import csv
    import matplotlib.pyplot as plt

    results_png = RUN_DIR / "results.png"
    print("Training curves:", results_png)

    # --- Read results.csv (one row per epoch; column names have extra spaces) ---
    with open(RUN_DIR / "results.csv", newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]

    def val(row, col):
        return float(row[col])

    # Best epoch = highest validation mAP@0.5:0.95 (this is what best.pt keeps)
    best = max(rows, key=lambda r: val(r, "metrics/mAP50-95(B)"))
    first, last = rows[0], rows[-1]
    epochs_trained = len(rows)

    print(f"\nEpochs trained : {epochs_trained} of {EPOCHS}"
          + ("  (stopped early by patience)" if epochs_trained < EPOCHS else ""))
    print(f"Training time  : {val(last, 'time') / 3600:.2f} hours")
    print(f"Best epoch     : {int(val(best, 'epoch'))}")

    # --- Small table: how the main numbers changed during training ---
    cols = [("train/box_loss", "train box"), ("val/box_loss", "val box"),
            ("metrics/precision(B)", "P"), ("metrics/recall(B)", "R"),
            ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]
    print(f"\n{'':<12}" + "".join(f"{name:>11}" for _, name in cols))
    for label, row in [("first epoch", first), ("best epoch", best), ("last epoch", last)]:
        print(f"{label:<12}" + "".join(f"{val(row, col):>11.4f}" for col, _ in cols))

    # --- Simple overfitting check: is the val loss at the end much worse than its lowest point? ---
    val_box = [val(r, "val/box_loss") for r in rows]
    min_val_box = min(val_box)
    rise = (val_box[-1] - min_val_box) / min_val_box * 100
    print(f"\nval box loss: lowest {min_val_box:.4f}, last {val_box[-1]:.4f} ({rise:+.1f}% vs lowest)")
    if rise > 10:
        print("-> val loss went back up noticeably: signs of overfitting.")
    else:
        print("-> val loss stayed near its lowest point: no sign of overfitting.")

    # --- mAP gain in the last 10 epochs: is it still improving or has it plateaued? ---
    if epochs_trained > 10:
        gain = val(last, "metrics/mAP50-95(B)") - val(rows[-11], "metrics/mAP50-95(B)")
        print(f"mAP50-95 change over the last 10 epochs: {gain:+.4f}"
              + ("  (plateau: more epochs would add little)" if gain < 0.01 else "  (still improving)"))

    if SHOW_PLOTS:
        img = plt.imread(str(results_png))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(img)
        ax.axis("off")
        plt.tight_layout()
        plt.show()  # close the window to continue
        plt.close(fig)

    return {
        "epochs_trained": epochs_trained,
        "best_epoch": int(val(best, "epoch")),
        "training_hours": round(val(last, "time") / 3600, 2),
        "val_precision": val(best, "metrics/precision(B)"),
        "val_recall": val(best, "metrics/recall(B)"),
        "val_mAP50": val(best, "metrics/mAP50(B)"),
        "val_mAP50-95": val(best, "metrics/mAP50-95(B)"),
    }


def show_image_file(path, title=None, figsize=(10, 8)):
    """Open a saved PNG in a window (only when SHOW_PLOTS is True)."""
    import matplotlib.pyplot as plt
    if not SHOW_PLOTS:
        return
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(plt.imread(str(path)))
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()  # close the window to continue
    plt.close(fig)


# ===========================================================================
# PART 7: Evaluate best.pt on the TEST split
# ---------------------------------------------------------------------------
# The validation set was used during training to pick best.pt, so its
# scores are slightly optimistic. The TEST set was never used for training
# decisions, so it gives the final numbers for the report.
#
# - Precision: of the boxes the model predicted, how many were real plates
#              (few false alarms -> high).
# - Recall:    of the real plates, how many the model found (few misses -> high).
# - F1:        one number balancing both: 2*P*R / (P + R).
# - mAP@0.5:   average precision when a box needs only 50% overlap (IoU)
#              with the true box.
# - mAP@0.5:0.95: averaged over stricter overlaps (IoU 0.5 to 0.95), so it
#              rewards TIGHT boxes (important for cropping plates for OCR).
# Ultralytics reports P and R at the confidence threshold with the best F1.
#
# Confusion matrix (one class): plates found (license -> license), plates
# missed (license -> background), false alarms (background -> license).
# PR curve: precision vs recall as the confidence threshold changes; the
# closer it hugs the top-right corner, the better.
# ===========================================================================
def evaluate_on_test(best_pt, device, batch):
    """Run validation on the test split. Returns (metrics dict, confusion matrix png, PR curve png)."""
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(YAML_PATH),
        split="test",           # use the test images, not valid
        imgsz=TRAIN_SIZE,
        batch=batch,
        device=device,
        workers=WORKERS,        # keep memory use low (default would start 8 workers)
        project=str(RUNS_DIR),
        name=f"{RUN_NAME}_test",
        exist_ok=True,
        plots=True,             # saves confusion matrix and PR curve
    )
    test_dir = Path(metrics.save_dir)

    precision = float(metrics.box.mp)  # mean precision
    recall = float(metrics.box.mr)     # mean recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    test_results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    print("\nTest set results")
    print("-" * 26)
    for name, value in test_results.items():
        print(f"{name:<12}{value:>14.4f}")

    # File is "BoxPR_curve.png" in newer Ultralytics versions, "PR_curve.png" in older ones
    confusion_png = test_dir / "confusion_matrix.png"
    pr_png = next(test_dir.glob("*PR_curve.png"))
    print("\nConfusion matrix:", confusion_png)
    print("PR curve        :", pr_png)
    show_image_file(confusion_png, "Confusion matrix (test)")
    show_image_file(pr_png, "Precision-Recall curve (test)")

    return test_results, confusion_png, pr_png


# ===========================================================================
# PART 8: Predictions on test images (including failures)
# ---------------------------------------------------------------------------
# We run the model on EVERY test image and compare its boxes with the
# ground truth:
# - a prediction is correct (TP) if it overlaps a not-yet-matched true
#   plate with IoU >= 0.5,
# - otherwise it's a FALSE POSITIVE (FP): it boxed something that isn't a plate,
# - true plates left unmatched are MISSES (FN).
# Then we show 8 images: up to 4 failures (a miss or an FP) and the rest
# correct. Green = ground truth, red = prediction with its confidence.
# Look for patterns in the failures: tiny/far plates, blur, night,
# odd angles, plate-like text on signs or trucks.
# ===========================================================================
CONF_THRESHOLD = 0.25  # ignore predictions below this confidence
IOU_MATCH = 0.5        # overlap needed to count as a correct detection


def box_iou(a, b):
    """IoU (intersection over union) of two boxes in [x1, y1, x2, y2] pixel format."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ground_truth_xyxy(lbl_path, img_w, img_h):
    """Ground-truth boxes of one image in pixel [x1, y1, x2, y2] format."""
    boxes = read_label_boxes(lbl_path) if lbl_path.exists() else []
    return [[(xc - w / 2) * img_w, (yc - h / 2) * img_h, (xc + w / 2) * img_w, (yc + h / 2) * img_h]
            for _, xc, yc, w, h in boxes]


def match_predictions(preds, confs, gts):
    """Greedy matching, most confident prediction first. Returns (tp, fp, fn)."""
    import numpy as np
    matched = set()  # indices of true boxes already used
    tp = fp = 0
    for i in np.argsort(-confs):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j not in matched:
                iou = box_iou(preds[i], g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
        if best_iou >= IOU_MATCH:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1
    fn = len(gts) - len(matched)
    return tp, fp, fn


def show_test_predictions(best_pt, device, n_show=8, max_failures=4, seed=42):
    """Predict all test images, count TP/FP/FN, and plot 8 examples incl. failures."""
    import random
    import cv2
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    test_img_dir = DATA_DIR / "test" / "images"
    test_lbl_dir = DATA_DIR / "test" / "labels"

    per_image = []
    # stream=True returns results one at a time instead of keeping all images in memory
    for r in model.predict(source=str(test_img_dir), imgsz=TRAIN_SIZE, conf=CONF_THRESHOLD,
                           device=device, stream=True, verbose=False):
        img_path = Path(r.path)
        img_h, img_w = r.orig_shape
        preds = r.boxes.xyxy.cpu().numpy()  # predicted boxes in pixels
        confs = r.boxes.conf.cpu().numpy()  # their confidence scores
        gts = ground_truth_xyxy(test_lbl_dir / f"{img_path.stem}.txt", img_w, img_h)
        tp, fp, fn = match_predictions(preds, confs, gts)
        per_image.append({"path": img_path, "preds": preds, "confs": confs, "gts": gts,
                          "tp": tp, "fp": fp, "fn": fn})

    failures = [x for x in per_image if x["fp"] > 0 or x["fn"] > 0]
    correct = [x for x in per_image if x["fp"] == 0 and x["fn"] == 0]
    failure_stats = {
        "conf_threshold": CONF_THRESHOLD,
        "iou_match": IOU_MATCH,
        "test_images": len(per_image),
        "true_plates": sum(len(x["gts"]) for x in per_image),
        "true_positives": sum(x["tp"] for x in per_image),
        "false_positives": sum(x["fp"] for x in per_image),
        "missed_plates": sum(x["fn"] for x in per_image),
        "images_with_miss": sum(1 for x in per_image if x["fn"] > 0),
        "images_with_false_positive": sum(1 for x in per_image if x["fp"] > 0),
    }
    print(f"Matching at conf >= {CONF_THRESHOLD} and IoU >= {IOU_MATCH}:")
    for k, v in failure_stats.items():
        print(f"  {k:<28}{v}")

    # --- Pick up to 4 failures, fill the rest with correct images ---
    random.seed(seed)
    n_fail = min(max_failures, len(failures))
    chosen = random.sample(failures, n_fail) + random.sample(correct, min(n_show - n_fail, len(correct)))
    if n_fail < 2:
        print(f"Only {n_fail} failure case(s) found at conf={CONF_THRESHOLD}.")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, item in zip(axes.flat, chosen):
        img = cv2.cvtColor(cv2.imread(str(item["path"])), cv2.COLOR_BGR2RGB)
        t = max(2, img.shape[1] // 300)
        for g in item["gts"]:  # ground truth = green
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 255, 0), t)
        for p, c in zip(item["preds"], item["confs"]):  # prediction = red, with confidence
            cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (255, 0, 0), t)
            cv2.putText(img, f"{c:.2f}", (int(p[0]), max(int(p[1]) - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if item["fn"] or item["fp"]:
            title, color = f"FAIL: {item['fn']} missed, {item['fp']} false pos.", "red"
        else:
            title, color = f"OK: {item['tp']} plate(s) found", "green"
        ax.imshow(img)
        ax.set_title(title, color=color, fontsize=11)
        ax.axis("off")
        print(f"  {title:<36} {item['path'].name}")
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")

    fig.suptitle("Green = ground truth, red = prediction (confidence)", fontsize=13)
    plt.tight_layout()
    chart_path = OUTPUTS_DIR / f"{RUN_NAME}_test_predictions.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    print("Prediction grid saved:", chart_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    return failure_stats


# ===========================================================================
# PART 9: Inference speed
# ---------------------------------------------------------------------------
# For a real system (e.g. a gate camera), speed matters as much as
# accuracy. We predict test images ONE AT A TIME (batch size 1, like a
# live video feed) and measure:
# - model time: preprocess (resize/normalize) + inference (the neural
#   network) + postprocess (filtering boxes), as timed by Ultralytics,
# - end-to-end time: wall-clock time per image INCLUDING reading the file.
# FPS = 1000 / (ms per image). The first predictions are slower (GPU
# warm-up), so 10 warm-up images are not counted.
# ===========================================================================
def measure_speed(best_pt, device, device_name, n_warmup=10):
    """Time single-image predictions on the test set. Returns a dict."""
    import time
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    test_images = list_images(DATA_DIR / "test" / "images")

    for p in test_images[:n_warmup]:  # warm-up, not timed
        model.predict(str(p), imgsz=TRAIN_SIZE, device=device, verbose=False)

    pre_ms = inf_ms = post_ms = 0.0
    start = time.perf_counter()
    for p in test_images:
        r = model.predict(str(p), imgsz=TRAIN_SIZE, device=device, verbose=False)[0]
        pre_ms += r.speed["preprocess"]
        inf_ms += r.speed["inference"]
        post_ms += r.speed["postprocess"]
    wall_s = time.perf_counter() - start

    n = len(test_images)
    model_ms = (pre_ms + inf_ms + post_ms) / n
    end_to_end_ms = wall_s * 1000 / n
    speed = {
        "device": device_name,
        "images": n,
        "preprocess_ms": round(pre_ms / n, 2),
        "inference_ms": round(inf_ms / n, 2),
        "postprocess_ms": round(post_ms / n, 2),
        "model_ms_per_image": round(model_ms, 2),
        "model_fps": round(1000 / model_ms, 1),
        "end_to_end_ms_per_image": round(end_to_end_ms, 2),
        "end_to_end_fps": round(1000 / end_to_end_ms, 1),
    }

    print(f"Device: {device_name}, {n} test images, batch size 1, imgsz {TRAIN_SIZE}\n")
    print(f"{'preprocess':<22}{speed['preprocess_ms']:>8.2f} ms")
    print(f"{'inference':<22}{speed['inference_ms']:>8.2f} ms")
    print(f"{'postprocess':<22}{speed['postprocess_ms']:>8.2f} ms")
    print(f"{'model total':<22}{model_ms:>8.2f} ms  ->  {1000 / model_ms:6.1f} FPS")
    print(f"{'end-to-end (w/ disk)':<22}{end_to_end_ms:>8.2f} ms  ->  {1000 / end_to_end_ms:6.1f} FPS")
    return speed


# ===========================================================================
# PART 10: Save outputs for the report
# ---------------------------------------------------------------------------
# Copies everything the report needs into plate_detection/outputs/:
# - <RUN_NAME>_best.pt: the trained detector (the OCR script loads this later)
# - <RUN_NAME>_results.png, _confusion_matrix.png, _PR_curve.png: figures
# - <RUN_NAME>_metrics.json: full summary (settings, dataset counts,
#   training summary, test metrics, error counts, speed)
# - metrics_summary.csv: ONE ROW PER RUN, so if you train yolo26s later,
#   both models end up in the same table for comparison.
# Parts 3, 4 and 8 already saved their figures there.
# ===========================================================================
def save_outputs(best_pt, device_name, batch, split_stats, training_summary,
                 test_results, failure_stats, speed, confusion_png, pr_png):
    """Copy weights/figures and write the JSON + CSV metrics summary."""
    import csv
    import json

    # --- Weights and figures ---
    shutil.copy(best_pt, OUTPUTS_DIR / f"{RUN_NAME}_best.pt")
    shutil.copy(RUN_DIR / "results.png", OUTPUTS_DIR / f"{RUN_NAME}_results.png")
    shutil.copy(confusion_png, OUTPUTS_DIR / f"{RUN_NAME}_confusion_matrix.png")
    shutil.copy(pr_png, OUTPUTS_DIR / f"{RUN_NAME}_PR_curve.png")

    # --- Full summary as JSON ---
    summary = {
        "run_name": RUN_NAME,
        "model": MODEL_WEIGHTS,
        "settings": {
            "imgsz": TRAIN_SIZE, "epochs_max": EPOCHS, "patience": PATIENCE,
            "seed": SEED, "batch": batch, "workers": WORKERS, "device": device_name,
        },
        "dataset": split_stats,
        "training": training_summary,
        "test_metrics": {k: round(v, 4) for k, v in test_results.items()},
        "test_error_analysis": failure_stats,
        "speed": speed,
    }
    json_path = OUTPUTS_DIR / f"{RUN_NAME}_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # --- One row per run in a CSV (replace this run's row if it already exists) ---
    csv_path = OUTPUTS_DIR / "metrics_summary.csv"
    row = {
        "run_name": RUN_NAME,
        "model": MODEL_WEIGHTS,
        "epochs_trained": training_summary["epochs_trained"],
        "best_epoch": training_summary["best_epoch"],
        "training_hours": training_summary["training_hours"],
        **{f"test_{k}": round(v, 4) for k, v in test_results.items()},
        "missed_plates": failure_stats["missed_plates"],
        "false_positives": failure_stats["false_positives"],
        "ms_per_image": speed["model_ms_per_image"],
        "fps": speed["model_fps"],
        "device": device_name,
    }
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run_name") != RUN_NAME]
    rows.append(row)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Saved to", OUTPUTS_DIR)
    for p in sorted(OUTPUTS_DIR.iterdir()):
        print(f"  {p.name:<45}{p.stat().st_size / 1024:>10.1f} KB")


# ===========================================================================
# Run everything in order
# ===========================================================================
def main():
    print("=" * 60)
    print("PART 1: Environment setup")
    print("=" * 60)
    device, device_name, batch = setup_environment()

    print("\n" + "=" * 60)
    print("PART 2: Fix data.yaml")
    print("=" * 60)
    class_names = fix_data_yaml()

    print("\n" + "=" * 60)
    print("PART 2b: Fix label files that mix polygons and boxes")
    print("=" * 60)
    fix_mixed_labels()

    print("\n" + "=" * 60)
    print("PART 3: Dataset sanity check")
    print("=" * 60)
    split_stats = dataset_sanity_check()

    print("\n" + "=" * 60)
    print("PART 4: Training samples with boxes")
    print("=" * 60)
    show_training_samples(class_names)

    print("\n" + "=" * 60)
    print("PART 5: Train the detector")
    print("=" * 60)
    best_pt = train_detector(device, batch)

    print("\n" + "=" * 60)
    print("PART 6: Training curves")
    print("=" * 60)
    training_summary = show_training_curves()

    print("\n" + "=" * 60)
    print("PART 7: Evaluate on the TEST split")
    print("=" * 60)
    test_results, confusion_png, pr_png = evaluate_on_test(best_pt, device, batch)

    print("\n" + "=" * 60)
    print("PART 8: Predictions on test images (incl. failures)")
    print("=" * 60)
    failure_stats = show_test_predictions(best_pt, device)

    print("\n" + "=" * 60)
    print("PART 9: Inference speed")
    print("=" * 60)
    speed = measure_speed(best_pt, device, device_name)

    print("\n" + "=" * 60)
    print("PART 10: Save outputs for the report")
    print("=" * 60)
    save_outputs(best_pt, device_name, batch, split_stats, training_summary,
                 test_results, failure_stats, speed, confusion_png, pr_png)

    print("\nAll parts finished.")


if __name__ == "__main__":
    main()
