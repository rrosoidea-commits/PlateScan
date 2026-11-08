"""
AI-based License Plate Recognition - Part 2: READING the plate number (OCR)
==========================================================================
Uses the plate detector trained in AI_PlateNumber.py to find plates, crops
each one, and reads its characters with a pretrained OCR model (no
training needed). Two OCR engines are supported (OCR_ENGINE setting):
  - "fast-plate-ocr": a small model trained ONLY on license plates (default)
  - "easyocr"       : a general-purpose text reader (the first version)

Pipeline:
    image -> [YOLO26 detector] -> plate box -> crop + clean up -> [OCR] -> "ABC1234"

Run from a terminal:
    python AI_PlateNumber_OCR.py

Like the detector script, everything is split into numbered parts
(functions) and main() at the bottom runs them in order.
"""

import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent  # the repository folder

DATA_DIR = ROOT / "data" / "License Plate Detection MERGED PROJECTS.v3-best-model.yolo26"
TEST_IMG_DIR = DATA_DIR / "test" / "images"

BASE_DIR = ROOT / "plate_detection"
DETECTOR_PT = ROOT / "models" / "yolo26n_640_e50_best.pt"  # trained detector (copy of Part 10's best.pt)
OCR_DIR = BASE_DIR / "ocr_outputs"   # everything this script saves goes here
CROPS_DIR = OCR_DIR / "crops"        # the cropped plates (for checking by eye)

# Your own photos: set to an image file or a folder, e.g.
#   INPUT_PATH = Path(r"C:\path\to\car.jpg")
# Leave as None to skip Part 8.
INPUT_PATH = None

# SHOW_PLOTS = True opens each chart in a window and PAUSES until you close it.
# Charts are always saved as PNG files in OCR_DIR either way.
SHOW_PLOTS = False

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DET_CONF = 0.25        # minimum detector confidence for a plate
PAD = 0.05             # extra margin around each plate box (5% of its size)

# Which OCR engine reads the plates: "fast-plate-ocr" or "easyocr".
# Results on 23 hand-labeled test crops (not the ones in the Part 7 manual check):
#   EasyOCR, first version (height 96, CLAHE, greedy)       :  1/23 exact,  56% characters
#   EasyOCR, tuned (median 3, height 160, beamsearch)       :  7/23 exact,  74% characters, ~60 ms/plate (GPU)
#   fast-plate-ocr, cct-xs-v2-global-model                  : 20/23 exact,  98% characters,  ~2 ms/plate (CPU)
OCR_ENGINE = "fast-plate-ocr"

# fast-plate-ocr settings
FPO_MODEL = "cct-xs-v2-global-model"  # trained on plates from ~65 countries

# EasyOCR settings (only used when OCR_ENGINE = "easyocr")
MEDIAN_KSIZE = 3       # median filter size, removes salt-and-pepper noise
OCR_HEIGHT = 160       # crops are enlarged to this height before OCR
OCR_DECODER = "beamsearch"  # "greedy" (faster) or "beamsearch" (more accurate)
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # characters a plate can have
MAIN_TEXT_RATIO = 0.75 # keep text at least 75% as tall as the biggest text (drops slogans)
SEED = 42


# ===========================================================================
# PART 1: Setup - install the OCR engine, load the detector and the reader
# ---------------------------------------------------------------------------
# - fast-plate-ocr: a tiny recognizer (CCT = Compact Convolutional
#   Transformer) trained only on license plate crops from ~65 countries.
#   It runs with ONNX Runtime on the CPU and takes ~2 ms per plate.
#   The first run downloads the model (a few MB).
# - EasyOCR (optional): general-purpose OCR on PyTorch (uses the GPU).
#   The first run downloads its English models (~100 MB).
# - The detector is the best.pt from the training script.
# ===========================================================================
def pip_install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def setup():
    """Install/load everything. Returns (detector, reader, device, device_name)."""
    import torch
    from ultralytics import YOLO

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print("Device   :", device_name)

    if not DETECTOR_PT.exists():
        raise FileNotFoundError(f"Detector not found: {DETECTOR_PT}\nRun AI_PlateNumber.py first.")
    detector = YOLO(str(DETECTOR_PT))
    print("Detector :", DETECTOR_PT)

    if OCR_ENGINE == "fast-plate-ocr":
        try:
            from fast_plate_ocr import LicensePlateRecognizer
        except ImportError:
            pip_install("fast-plate-ocr[onnx]")  # [onnx] = CPU version of ONNX Runtime
            from fast_plate_ocr import LicensePlateRecognizer
        import onnxruntime
        onnxruntime.set_default_logger_severity(3)  # hide harmless ONNX Runtime warnings
        reader = LicensePlateRecognizer(FPO_MODEL, device="cpu")
        print(f"OCR      : fast-plate-ocr ({FPO_MODEL})")
    elif OCR_ENGINE == "easyocr":
        try:
            import easyocr
        except ImportError:
            pip_install("easyocr")
            import easyocr
        # 'en' = English letters and digits. gpu=True runs OCR on the GPU.
        reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        print(f"OCR      : EasyOCR {easyocr.__version__}")
    else:
        raise ValueError(f"Unknown OCR_ENGINE: {OCR_ENGINE}")

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    return detector, reader, device, device_name


# ===========================================================================
# PART 2: Detect plates and crop them
# ---------------------------------------------------------------------------
# The detector gives a box (x1, y1, x2, y2) per plate. We add a small
# margin (PAD) so characters touching the box edge are not cut off, then
# cut that region out of the image.
# ===========================================================================
def detect_plates(detector, img_bgr, device):
    """Return a list of plates: {'box': [x1,y1,x2,y2], 'det_conf': float, 'crop': image}."""
    result = detector.predict(img_bgr, imgsz=640, conf=DET_CONF, device=device, verbose=False)[0]
    img_h, img_w = img_bgr.shape[:2]

    plates = []
    for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        pad_x, pad_y = (x2 - x1) * PAD, (y2 - y1) * PAD
        # add the margin, but stay inside the image
        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(img_w, x2 + pad_x))
        y2 = int(min(img_h, y2 + pad_y))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue  # too small to read anything
        plates.append({"box": [x1, y1, x2, y2], "det_conf": float(conf),
                       "crop": img_bgr[y1:y2, x1:x2].copy()})
    return plates


# ===========================================================================
# PART 3: Prepare the crop for the OCR engine
# ---------------------------------------------------------------------------
# fast-plate-ocr: the crop is only converted from OpenCV's BGR color order
# to RGB. The model resizes it to 128x64 itself, and it was trained on
# real (noisy, blurry, tilted) plates, so extra filtering didn't help
# in our test (98% characters with or without a median filter).
#
# EasyOCR works best on large, clean, grayscale text:
# 1. Grayscale: color doesn't help read characters.
# 2. Median filter (3x3): replaces each pixel with the median of its
#    neighbors. This removes the salt-and-pepper noise (random black and
#    white dots) that Roboflow added to many images, while keeping
#    character edges sharp. It gave the biggest accuracy gain.
# 3. Resize to a fixed height (160 px): small plates get enlarged so
#    EasyOCR's text finder can see the characters (cubic interpolation
#    keeps edges smooth).
# We also tried CLAHE (local contrast boost): it made results WORSE here,
# because it also boosts the noise, so it is not used.
# ===========================================================================
def preprocess_plate(crop_bgr):
    """Prepare a plate crop for the selected OCR engine."""
    import cv2

    if OCR_ENGINE == "fast-plate-ocr":
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # EasyOCR: grayscale + denoise + resize
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, MEDIAN_KSIZE)
    scale = OCR_HEIGHT / gray.shape[0]
    new_w = max(1, int(gray.shape[1] * scale))
    return cv2.resize(gray, (new_w, OCR_HEIGHT), interpolation=cv2.INTER_CUBIC)


# ===========================================================================
# PART 4: Read the characters
# ---------------------------------------------------------------------------
# fast-plate-ocr reads the whole plate in one step. It has 10 character
# "slots" and predicts one character (0-9, A-Z) or "empty" for each slot,
# so it needs no separate text-finding step and ignores small text like
# state names and slogans on its own. It returns:
#   plate      - the text, e.g. "ABC1234"
#   char_probs - confidence of each slot; we average the slots that hold
#                real characters to get one confidence for the plate.
#
# EasyOCR finds text pieces in the crop and reads each one, returning
# (box, text, confidence). Plates often have extra small text (state name,
# slogans, dealer frames), so we:
# - keep only text pieces at least 75% as tall as the biggest one (the
#   registration number is the largest text on a plate),
# - sort them top-to-bottom, then left-to-right (handles 2-row plates),
# - join them and keep only A-Z and 0-9.
# The plate confidence is the average OCR confidence of the kept pieces.
# ===========================================================================
def clean_text(text):
    """Uppercase and keep only letters and digits."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def read_plate(reader, plate_img):
    """OCR one prepared plate image. Returns (text, ocr_conf)."""
    if OCR_ENGINE == "fast-plate-ocr":
        pred = reader.run(plate_img, return_confidence=True)[0]
        text = clean_text(pred.plate)
        if not text:
            return "", 0.0
        conf = float(pred.char_probs[:len(text)].mean())  # only the slots with characters
        return text, conf

    results = reader.readtext(plate_img, allowlist=ALLOWLIST, detail=1, paragraph=False,
                              decoder=OCR_DECODER)
    if not results:
        return "", 0.0

    pieces = []
    for points, text, conf in results:
        ys = [p[1] for p in points]
        xs = [p[0] for p in points]
        pieces.append({"text": clean_text(text), "conf": float(conf),
                       "height": max(ys) - min(ys), "x": min(xs), "y_center": (min(ys) + max(ys)) / 2})

    max_h = max(p["height"] for p in pieces)
    main = [p for p in pieces if p["height"] >= MAIN_TEXT_RATIO * max_h and p["text"]]
    if not main:
        return "", 0.0

    # Same row = y-centers within half a character height; then left to right
    main.sort(key=lambda p: (round(p["y_center"] / (0.5 * max_h)), p["x"]))
    text = "".join(p["text"] for p in main)
    conf = sum(p["conf"] for p in main) / len(main)
    return text, conf


# ===========================================================================
# PART 5: Full pipeline on one image + demo on 8 test images
# ---------------------------------------------------------------------------
# recognize_image() chains Parts 2-4 and times the detector and the OCR
# separately. The demo draws each detected plate with the text read from
# it, and a second figure shows each crop next to its cleaned-up version,
# so you can see what the OCR actually received.
# ===========================================================================
def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)


def recognize_image(detector, reader, img_bgr, device):
    """Detect + read every plate in one image. Returns (plates, det_ms, ocr_ms)."""
    import time

    t0 = time.perf_counter()
    plates = detect_plates(detector, img_bgr, device)
    t1 = time.perf_counter()
    for plate in plates:
        plate["clean"] = preprocess_plate(plate["crop"])
        plate["text"], plate["ocr_conf"] = read_plate(reader, plate["clean"])
    t2 = time.perf_counter()
    return plates, (t1 - t0) * 1000, (t2 - t1) * 1000


def draw_plates(img_bgr, plates):
    """Draw boxes and the read text on a copy of the image (RGB, for matplotlib)."""
    import cv2

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = max(2, img.shape[1] // 300)
    for p in plates:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), t)
        label = p["text"] or "?"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        ty = max(y1 - 6, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), (255, 0, 0), -1)  # label background
        cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def demo_on_test_images(detector, reader, device, n_images=8):
    """Run the pipeline on random test images, plot results and crops."""
    import random
    import cv2
    import matplotlib.pyplot as plt

    random.seed(SEED)
    samples = random.sample(list_images(TEST_IMG_DIR), n_images)

    all_crops = []  # (crop, cleaned, text, conf) for the second figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, img_path in zip(axes.flat, samples):
        img = cv2.imread(str(img_path))
        plates, _, _ = recognize_image(detector, reader, img, device)
        ax.imshow(draw_plates(img, plates))
        texts = ", ".join(p["text"] or "?" for p in plates) or "no plate found"
        ax.set_title(texts, fontsize=12)
        ax.axis("off")
        print(f"  {img_path.name[:45]:<46} -> {texts}")
        for p in plates:
            all_crops.append((p["crop"], p["clean"], p["text"], p["ocr_conf"]))

    plt.tight_layout()
    path = OCR_DIR / "ocr_demo_images.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("Saved:", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

    # --- Crops: original vs cleaned-up, with the text read ---
    crops = all_crops[:8]
    if crops:
        fig, axes = plt.subplots(len(crops), 2, figsize=(8, 1.6 * len(crops)), squeeze=False)
        for row, (crop, clean, text, conf) in zip(axes, crops):
            row[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            row[0].set_title("detector crop", fontsize=9)
            row[1].imshow(clean, cmap="gray" if clean.ndim == 2 else None)
            row[1].set_title(f"OCR input -> '{text or '?'}' (conf {conf:.2f})", fontsize=9)
            for a in row:
                a.axis("off")
        plt.tight_layout()
        path = OCR_DIR / "ocr_demo_crops.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print("Saved:", path)
        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)


# ===========================================================================
# PART 6: Run on the whole test set
# ---------------------------------------------------------------------------
# Reads every plate in the 593 test images and saves:
# - ocr_results.csv: one row per detected plate (image, box, detector
#   confidence, text, OCR confidence),
# - crops/: each plate crop as a small image file (for checking by eye).
# It also measures the speed of the full pipeline (detection + OCR).
#
# NOTE: the dataset has boxes but NO plate-number labels, so we can't
# compute OCR accuracy automatically. Part 7 handles that.
# ===========================================================================
def run_on_test_set(detector, reader, device, device_name):
    """OCR every test image; write CSV + crops; print a summary. Returns the list of rows."""
    import csv
    import cv2

    images = list_images(TEST_IMG_DIR)
    rows, det_times, ocr_times = [], [], []

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        plates, det_ms, ocr_ms = recognize_image(detector, reader, img, device)
        det_times.append(det_ms)
        ocr_times.append(ocr_ms)
        for k, p in enumerate(plates):
            crop_name = f"{img_path.stem[:60]}_plate{k}.png"
            cv2.imwrite(str(CROPS_DIR / crop_name), p["crop"])
            rows.append({
                "image": img_path.name,
                "plate": k,
                "crop_file": crop_name,
                "det_conf": round(p["det_conf"], 3),
                "x1": p["box"][0], "y1": p["box"][1], "x2": p["box"][2], "y2": p["box"][3],
                "text": p["text"],
                "ocr_conf": round(p["ocr_conf"], 3),
                "n_chars": len(p["text"]),
            })
        if i % 100 == 0:
            print(f"  {i}/{len(images)} images done")

    csv_path = OCR_DIR / "ocr_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- Summary ---
    n = len(images)
    n_plates = len(rows)
    read = [r for r in rows if r["text"]]
    typical = [r for r in read if 4 <= r["n_chars"] <= 8]  # most plates have 4-8 characters
    avg_det = sum(det_times[5:]) / max(1, n - 5)            # skip 5 warm-up images
    avg_ocr = sum(ocr_times[5:]) / max(1, n - 5)

    print(f"\nTest images              : {n}")
    print(f"Plates detected          : {n_plates}")
    print(f"Plates with text read    : {len(read)} ({100 * len(read) / max(1, n_plates):.1f}%)")
    print(f"  ...with 4-8 characters : {len(typical)} ({100 * len(typical) / max(1, n_plates):.1f}%)")
    if read:
        print(f"Average OCR confidence   : {sum(r['ocr_conf'] for r in read) / len(read):.3f}")
    print(f"\nSpeed on {device_name} (per image):")
    print(f"  detection   : {avg_det:7.1f} ms")
    print(f"  OCR         : {avg_ocr:7.1f} ms")
    print(f"  full pipeline: {avg_det + avg_ocr:6.1f} ms  ->  {1000 / (avg_det + avg_ocr):.1f} FPS")
    print("Saved:", csv_path)
    print("Crops:", CROPS_DIR)
    return rows


# ===========================================================================
# PART 7: Measure OCR accuracy by hand
# ---------------------------------------------------------------------------
# Because the dataset has no plate-number labels, we check a random sample
# of 50 plates by hand:
# 1. The first run creates ocr_outputs/manual_check.csv with columns
#    crop_file, ocr_text, correct_text (empty).
# 2. Open each crop in ocr_outputs/crops/, type the real plate number in
#    "correct_text" (letters/digits only). Type "-" if it's unreadable
#    even for you (those are skipped).
# 3. Run the script again: this part then compares your answers with the
#    CURRENT OCR results (from ocr_results.csv, so it stays correct if you
#    change the OCR settings later) and computes
#    - plate accuracy: % of plates read EXACTLY right,
#    - character accuracy: 1 - (character edits needed / true length),
#      using the edit (Levenshtein) distance.
# The file is never overwritten once it exists, so your typing is safe.
# ===========================================================================
def edit_distance(a, b):
    """Minimum number of insert/delete/replace steps to turn string a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                # delete
                           cur[j - 1] + 1,             # insert
                           prev[j - 1] + (ca != cb)))  # replace (free if same)
        prev = cur
    return prev[-1]


def manual_accuracy_check(rows, sample_size=50):
    """Create the manual-check sheet, or score it if it has been filled in."""
    import csv
    import random

    sheet = OCR_DIR / "manual_check.csv"

    if not sheet.exists():
        random.seed(SEED)
        sample = random.sample(rows, min(sample_size, len(rows)))
        with open(sheet, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["crop_file", "ocr_text", "correct_text"])
            for r in sample:
                writer.writerow([r["crop_file"], r["text"], ""])
        print(f"Created {sheet}")
        print(f"-> Open the crops in {CROPS_DIR}, fill in 'correct_text', then run this script again.")
        return None

    with open(sheet, newline="", encoding="utf-8") as f:
        filled = [r for r in csv.DictReader(f) if r["correct_text"].strip() not in ("", "-")]
    if not filled:
        print(f"{sheet.name} exists but 'correct_text' is still empty. Fill it in and run again.")
        return None

    # Use the text from THIS run (the sheet's ocr_text column may be from an older run)
    current_text = {r["crop_file"]: r["text"] for r in rows}

    exact = 0
    char_errors = 0
    char_total = 0
    print(f"{'OCR':<12}{'correct':<12}{'edits':>6}")
    for r in filled:
        ocr = clean_text(current_text.get(r["crop_file"], r["ocr_text"]))
        truth = clean_text(r["correct_text"])
        d = edit_distance(ocr, truth)
        exact += (d == 0)
        char_errors += d
        char_total += len(truth)
        if d:
            print(f"{ocr or '(empty)':<12}{truth:<12}{d:>6}")

    accuracy = {
        "plates_checked": len(filled),
        "plate_accuracy": exact / len(filled),
        "character_accuracy": max(0.0, 1 - char_errors / max(1, char_total)),
    }
    print(f"\nPlates checked      : {accuracy['plates_checked']}")
    print(f"Plate accuracy      : {100 * accuracy['plate_accuracy']:.1f}%  (read exactly right)")
    print(f"Character accuracy  : {100 * accuracy['character_accuracy']:.1f}%")
    return accuracy


# ===========================================================================
# PART 8: Try your own images
# ---------------------------------------------------------------------------
# Set INPUT_PATH at the top to one photo or a folder of photos. Each one
# is saved to ocr_outputs/my_images/ with the plates and text drawn on it.
# ===========================================================================
def run_on_my_images(detector, reader, device):
    import cv2

    path = Path(INPUT_PATH)
    images = [path] if path.is_file() else list_images(path)
    out_dir = OCR_DIR / "my_images"
    out_dir.mkdir(exist_ok=True)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  could not read {img_path.name}, skipped")
            continue
        plates, det_ms, ocr_ms = recognize_image(detector, reader, img, device)
        texts = ", ".join(f"{p['text'] or '?'} ({p['ocr_conf']:.2f})" for p in plates) or "no plate found"
        print(f"  {img_path.name}: {texts}  [{det_ms + ocr_ms:.0f} ms]")
        out = cv2.cvtColor(draw_plates(img, plates), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_ocr.jpg"), out)
    print("Saved to", out_dir)


# ===========================================================================
# Run everything in order
# ===========================================================================
def main():
    def header(title):
        print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)

    header("PART 1: Setup")
    detector, reader, device, device_name = setup()

    header("PARTS 2-5: Detect, crop, clean, read - demo on 8 test images")
    demo_on_test_images(detector, reader, device)

    header("PART 6: Run on the whole test set")
    rows = run_on_test_set(detector, reader, device, device_name)

    header("PART 7: OCR accuracy (manual check)")
    manual_accuracy_check(rows)

    if INPUT_PATH:
        header("PART 8: Your own images")
        run_on_my_images(detector, reader, device)

    print("\nAll parts finished. Outputs in", OCR_DIR)


if __name__ == "__main__":
    main()
