# AI-Based License Plate Recognition — Results & Analysis

> **Companion to `PROJECT_DOCUMENTATION.md`.** This document presents the **measured results** of the license plate detector (YOLO26n) and the plate reader (OCR), with charts, tables, and an interpretation of each. All figures are generated from the project's own outputs in `plate_detection/`.

---

## Contents
1. [Key Results at a Glance](#1-key-results-at-a-glance)
2. [Dataset Analysis](#2-dataset-analysis)
3. [Training Results](#3-training-results)
4. [Detector Performance on the Test Set](#4-detector-performance-on-the-test-set)
5. [Error Analysis](#5-error-analysis)
6. [Inference Speed](#6-inference-speed)
7. [Plate Reading (OCR) Results](#7-plate-reading-ocr-results)
8. [End-to-End System Estimate](#8-end-to-end-system-estimate)
9. [Key Findings & Recommendations](#9-key-findings--recommendations)
10. [Appendix: Figures & Reproducibility](#10-appendix-figures--reproducibility)

---

## 1. Key Results at a Glance

| Stage | Metric | Result |
|---|---|---|
| **Detector (test, 593 images)** | Precision | **0.985** |
| | Recall | **0.926** |
| | F1 score | **0.955** |
| | mAP@0.5 | **0.946** |
| | mAP@0.5:0.95 | **0.792** |
| | Plates found (conf ≥ 0.25) | **699 / 743** (44 missed, 32 false positives) |
| **Detector speed** | Time per image (RTX 2050, batch 1) | **15.4 ms → 65 FPS** |
| **OCR (fast-plate-ocr)** | Plates read exactly (23 hand-labeled crops) | **20 / 23 (87%)** |
| | Characters correct | **98.0%** |
| | OCR time per plate | **~3 ms (CPU)** |
| **Full pipeline** | Detect + read | **23.8 ms → 42 FPS** |
| **Training** | Epochs / best epoch / time | 50 / **47** / 4.42 h |

> **Bottom line:** the detector is accurate and fast; its weak spot is **plates smaller than 20 px**. The plate-specific OCR reads readable plates almost perfectly and is ~30× faster than EasyOCR; its accuracy also falls sharply on very small plates.

---

## 2. Dataset Analysis

![Dataset: images per split and plate size distribution](plate_detection/report_figures/fig_dataset.png)

| Split | Images | Share | Plates | Missing labels | Empty labels | Plates < 20 px | Plates ≥ 160 px |
|---|---|---|---|---|---|---|---|
| Train | 15,160 | 89.7% | 18,323 | 0 | 8 | 1.4% | 38.7% |
| Valid | 1,155 | 6.8% | 1,403 | 0 | 0 | 0.8% | 32.4% |
| Test | 593 | 3.5% | 743 | 0 | 2 | **3.6%** | 31.8% |
| **Total** | **16,908** | 100% | **20,469** | 0 | 10 | 1.4% | — |

**Interpretation**
- **Labels are complete.** Every image has a label file; the 10 empty ones are background images (no plate), which is acceptable and even helpful for learning what is *not* a plate.
- **Most plates are medium-sized.** The median plate is **118 px** tall at 640 px resolution; only **292 plates (1.4%)** are under 20 px.
- **The spike at ~620–640 px** comes from close-up images where the plate fills the whole frame (a sub-collection of the merged dataset).
- **The test set is harder than training on small plates:** 3.6% of its plates are under 20 px vs. 1.4% in train and 0.8% in validation. This explains part of the validation→test recall drop in Section 4.
- **797 label files (4.7%) mixed polygon and box rows** and would have been silently dropped by the training framework; they were repaired before training (see `PROJECT_DOCUMENTATION.md`).

**Sample training images with ground-truth boxes**

![Six random training images with their labels](plate_detection/outputs/train_samples.png)

The boxes fit the plates tightly. Roboflow's offline augmentations are visible (salt-and-pepper noise, rotation, a 4-image mosaic), and one image is synthetic (`DALL-E` in its filename).

---

## 3. Training Results

### 3.1 Loss curves

![Training and validation losses per epoch](plate_detection/report_figures/fig_training_losses.png)

| Loss | Epoch 1 (train → val) | Epoch 50 (train → val) | Trend |
|---|---|---|---|
| Box loss | 1.045 → 0.976 | 0.535 → 0.658 | Both falling; val at its minimum at the end |
| Classification loss | 1.68 → 0.57 | 0.20 → 0.26 | Both falling |
| L1 (coordinate) loss | 0.0149 → 0.0196 | 0.0074 → 0.0111 | Both falling |

**Interpretation**
- **No overfitting.** Validation box loss reached its **lowest value (0.658) in the final epoch**; a model that overfits shows validation loss turning upward while training loss keeps falling.
- **The sharp drop at epoch 41 (dashed line) is expected.** Mosaic augmentation is switched off for the last 10 epochs (`close_mosaic`), so training images suddenly become easier — train box loss fell from 0.706 to 0.628 in one epoch. Validation loss did not jump, confirming this is an augmentation effect, not instability.
- **The early bump (epochs 2–4)** is the learning-rate warm-up phase and is normal.
- **The train/val gap is small and stable**, which indicates good generalization to unseen images of the same distribution.

### 3.2 Validation metrics

![Validation precision, recall and mAP per epoch](plate_detection/report_figures/fig_training_metrics.png)

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 1 | 0.961 | 0.904 | 0.947 | 0.687 |
| 10 | — | — | — | 0.740 |
| 20 | — | — | — | 0.783 |
| 30 | — | — | — | 0.804 |
| 40 | — | — | — | 0.807 |
| **47 (best.pt)** | **0.976** | **0.956** | **0.974** | **0.816** |
| 50 | 0.976 | 0.954 | 0.974 | 0.814 |

**Interpretation**
- **Transfer learning gave a strong start:** after one epoch the COCO-pretrained model already reached mAP@0.5 = 0.947.
- **Training mainly improved recall and box tightness.** Precision was high from the start (0.96 → 0.98); recall rose **0.904 → 0.956** and mAP@0.5:0.95 rose **0.687 → 0.816** — the model learned to find harder plates and to fit boxes more tightly.
- **The model has plateaued.** mAP@0.5:0.95 gained only **+0.007** over the last 10 epochs. Training longer would add little; a larger model or better data would help more.
- **Early stopping never triggered** (patience 15) because small improvements continued until epoch 47.

<details>
<summary>Original Ultralytics training plot (all 10 curves)</summary>

![Ultralytics results.png](plate_detection/outputs/yolo26n_640_e50_results.png)
</details>

---

## 4. Detector Performance on the Test Set

### 4.1 Validation vs. test

![Validation vs test metrics](plate_detection/report_figures/fig_val_vs_test.png)

| Metric | Validation (best epoch) | Test | Change |
|---|---|---|---|
| Precision | 0.976 | **0.985** | +0.009 |
| Recall | 0.956 | **0.926** | −0.030 |
| F1 | 0.966 | **0.955** | −0.011 |
| mAP@0.5 | 0.974 | **0.946** | −0.028 |
| mAP@0.5:0.95 | 0.816 | **0.792** | −0.024 |

**Interpretation**
- **Precision held up (even rose); recall dropped by 3 points.** On new images the model stays reliable about *what it reports*, but it misses more plates.
- The drop is consistent with the test set's **higher share of tiny plates** (3.6% vs 0.8% in validation, Section 2) — Section 5 shows that tiny plates are exactly where recall collapses.
- A gap of 2–3 points between validation and test is normal: validation was used to pick `best.pt`, so its scores are slightly optimistic.

### 4.2 Precision–recall and F1 curves

| PR curve | F1–confidence curve |
|---|---|
| ![PR curve](plate_detection/runs/yolo26n_640_e50_test/BoxPR_curve.png) | ![F1 curve](plate_detection/runs/yolo26n_640_e50_test/BoxF1_curve.png) |

**Interpretation**
- **The PR curve hugs the top-right corner** (area = mAP@0.5 = 0.946): high precision is maintained across almost the whole recall range, falling only at the very end, where the remaining plates are the tiny/hard ones.
- **F1 peaks at 0.95 at confidence 0.608** and stays near its peak across a **wide plateau (~0.2 to ~0.75)**. The model is therefore **not sensitive to the exact threshold** — a useful property for deployment.
- Ultralytics reports the headline precision (0.985) and recall (0.926) **at this F1-optimal threshold (0.608)**.

### 4.3 Effect of the confidence threshold

![Precision, recall and F1 vs confidence threshold](plate_detection/report_figures/fig_threshold_sweep.png)

| Threshold | Correct (TP) | False positives | Missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0.01 | 712 | 746 | 31 | 0.488 | 0.958 | 0.647 |
| 0.05 | 706 | 201 | 37 | 0.778 | 0.950 | 0.856 |
| 0.10 | 702 | 104 | 41 | 0.871 | 0.945 | 0.906 |
| **0.25** (used in this project) | **699** | **32** | **44** | **0.956** | **0.941** | **0.948** |
| 0.40 | 694 | 17 | 49 | 0.976 | 0.934 | 0.955 |
| 0.50 | 692 | 13 | 51 | 0.982 | 0.931 | **0.956** |
| 0.60 | 684 | 7 | 59 | 0.990 | 0.921 | 0.954 |
| 0.70 | 676 | 7 | 67 | 0.990 | 0.910 | 0.948 |
| 0.80 | 640 | 6 | 103 | 0.991 | 0.861 | 0.922 |

*(Computed with this project's own matching: IoU ≥ 0.5, one prediction per true plate.)*

**Interpretation**
- **Raising the threshold trades misses for false alarms.** From 0.25 to 0.6, false positives fall from 32 to **7**, while misses rise only from 44 to 59.
- **Recommended operating points:**
  - **0.4–0.5** for a balanced system (best F1 ≈ 0.955),
  - **0.6+** when false alarms are costly (e.g., automated fines),
  - **0.1–0.25** when missing a plate is costly (e.g., searching for a stolen vehicle), with a human reviewing results.

### 4.4 Confusion matrix

![Confusion matrix](plate_detection/runs/yolo26n_640_e50_test/confusion_matrix.png)

| | True: license | True: background |
|---|---|---|
| **Predicted: license** | 717 | 2,174 |
| **Predicted: background** | 26 | — |

**Interpretation — read with care**
- **717 of 743 plates (96.5%)** were matched by some prediction; **26** were missed even at Ultralytics' near-zero threshold (with its slightly looser IoU 0.45 matching).
- **The 2,174 "background → license" entries are not real false alarms.** Ultralytics builds this matrix from validation run at a *near-zero* confidence threshold (0.001), so it counts thousands of very low-confidence guesses that a deployed system would discard. At the project's threshold of 0.25 there are only **32** false positives, and at 0.6 only **7** (Section 4.3).
- The "background/background" cell is empty by definition: in object detection there is no countable set of "correctly ignored background regions".

---

## 5. Error Analysis

### 5.1 Outcome of every test plate

![Correct, missed and false positive counts](plate_detection/report_figures/fig_error_breakdown.png)

| Outcome (conf ≥ 0.25, IoU ≥ 0.5) | Count | Images affected |
|---|---|---|
| ✔ Correct detections | **699** of 743 plates (94.1%) | — |
| ✖ Missed plates | 44 | 25 of 593 images |
| ! False positives | 32 | 22 of 593 images |

### 5.2 Where the detector fails: plate size

![Recall by plate size and confidence of correct vs false detections](plate_detection/report_figures/fig_recall_by_size_and_confidence.png)

| Plate height (at 640 px) | Plates in test set | Found | Recall |
|---|---|---|---|
| **< 20 px** | 27 | 5 | **19%** |
| 20–40 px | 26 | 21 | 81% |
| 40–80 px | 218 | 210 | 96% |
| 80–160 px | 236 | 230 | 97% |
| ≥ 160 px | 236 | 233 | **99%** |

**Interpretation**
- **Plate size is the dominant failure factor.** Recall is **96–99% for plates ≥ 40 px** but only **19% below 20 px**. The 27 tiny plates alone account for **22 of the 44 misses (50%)**, although they are just 3.6% of test plates.
- **Confidence separates correct and false detections well.** Correct detections have a median confidence of **0.90**; false positives a median of **0.44**. Most false positives therefore disappear with a slightly higher threshold (Section 4.3).

### 5.3 Visual failure cases

![Test predictions including failures (green = ground truth, red = prediction)](plate_detection/outputs/yolo26n_640_e50_test_predictions.png)

| Failure type | Example | Real problem? |
|---|---|---|
| **Tiny, distant plates** in crowded traffic | 4 missed, 7 false positives in one street scene | **Yes** — genuine model limitation; some "false positives" may be real but unlabeled plates |
| **Different box conventions** | Model boxes the whole plate (incl. frame text); the label boxes only the characters | **No** — the plate was found, but IoU < 0.5 counts it as 1 miss + 1 false positive |
| **Plates cut off at the image edge** | Partial plate boxed at a different size than the label | Partly — ambiguous ground truth |
| **Plates inside mosaic tiles** | Small partial plates in 4-image collages | Partly — artificial augmentation artifact |

**Interpretation:** a meaningful share of the counted errors are **annotation-style disagreements rather than detection failures**. This also lowers mAP@0.5:0.95 (0.792), which rewards exact box agreement.

---

## 6. Inference Speed

![Detector time breakdown and full pipeline timing](plate_detection/report_figures/fig_speed.png)

**Detector only** (593 test images, one at a time, NVIDIA RTX 2050 laptop GPU):

| Step | Time per image |
|---|---|
| Preprocess (resize, normalize) | 1.2 ms |
| Inference (neural network) | 12.7 ms |
| Postprocess (box filtering) | 1.5 ms |
| **Model total** | **15.4 ms → 65 FPS** |
| Including image loading from disk | 17.8 ms → 56 FPS |

**Full pipeline** (detect + read every plate):

| OCR engine | Detection | OCR | Total | Throughput |
|---|---|---|---|---|
| EasyOCR (GPU) | 17.6 ms | 41.3 ms | 58.9 ms | 17 FPS |
| **fast-plate-ocr (CPU)** | 20.8 ms | **3.0 ms** | **23.8 ms** | **42 FPS** |

**Interpretation**
- **Both the detector and the full pipeline exceed real-time video rates** (25–30 FPS) on an entry-level laptop GPU.
- **Inference dominates detector time (82%)**; pre- and post-processing are minor.
- **Switching OCR engines changed the bottleneck:** with EasyOCR, reading took 70% of the time; with fast-plate-ocr, reading takes just 13%, even though it runs on the CPU.
- Timings vary by a few percent between runs (an earlier run measured 16.1 ms / 62 FPS for the detector).

---

## 7. Plate Reading (OCR) Results

### 7.1 OCR engine comparison (23 hand-labeled crops)

![OCR accuracy by version](plate_detection/report_figures/fig_ocr_versions.png)

| Version | Preprocessing | Plates exact | Characters correct | Time / plate |
|---|---|---|---|---|
| EasyOCR v1 | Grayscale, CLAHE contrast, 96 px | 1 / 23 (4%) | 55.8% | — |
| EasyOCR v2 | Grayscale, median filter, 160 px, beam search | 7 / 23 (30%) | 74.1% | ~60 ms (GPU) |
| **fast-plate-ocr** `cct-xs-v2-global-model` | None (RGB crop) | **20 / 23 (87%)** | **98.0%** | **~2 ms (CPU)** |
| fast-plate-ocr `cct-s-v2-global-model` | Median filter | 20 / 23 | 97.3% | ~15 ms |
| fast-plate-ocr `global-plates-mobile-vit-v2` | Median filter | 14 / 23 | 85.0% | ~3 ms |

**Per-plate results (truth → EasyOCR v2 → fast-plate-ocr)**

| Truth | EasyOCR v2 | fast-plate-ocr | | Truth | EasyOCR v2 | fast-plate-ocr |
|---|---|---|---|---|---|---|
| AMA9460 | AHA9460 | ✅ AMA9460 | | MCKILLA | ✅ MCKILLA | ✅ MCKILLA |
| DUDE | DF | ✅ DUDE | | WACKI | MACKI | ✅ WACKI |
| GOGIRLL | GO6IRLLT | ✅ GOGIRLL | | BRDE633 | BRUEO | ❌ BRBE633 |
| SAA7877Y | S17877V | ✅ SAA7877Y | | AFR2011 | AFR20 | ✅ AFR2011 |
| ROCKESQ | ✅ ROCKESQ | ❌ ROCKESO | | WEBILLU | BILU | ✅ WEBILLU |
| MUDDY98 | FUDA | ✅ MUDDY98 | | SOUP4U | SOUPAU | ✅ SOUP4U |
| 7777777 | 7711777 | ✅ 7777777 | | 21184 | 1521184 | ❌ 221184 |
| P326361 | ✅ P326361 | ✅ P326361 | | MELLOW | MELFLON | ✅ MELLOW |
| FR17BEE | FRLZBEE | ✅ FR17BEE | | UNME2 | ✅ UNME2 | ✅ UNME2 |
| GGP297 | ✅ GGP297 | ✅ GGP297 | | TGVUP | ICHUPI | ✅ TGVUP |
| MCKILLA | ✅ MCKILLA | ✅ MCKILLA | | KUPCAKE | ✅ KUPCAKE | ✅ KUPCAKE |
| SOUP4U | SOUPQU | ✅ SOUP4U | | | | |

**Interpretation**
- **A plate-specific model beats tuned general OCR by a wide margin:** 87% vs 30% of plates exact. EasyOCR's typical errors — look-alike characters (M→H, 4→A), dropped characters, extra frame text — largely disappear.
- **Preprocessing mattered for EasyOCR, not for fast-plate-ocr.** The median filter (removing salt-and-pepper noise) was EasyOCR's biggest gain; CLAHE contrast enhancement *hurt* because it amplified the noise. fast-plate-ocr scored 98.0% with or without filtering.
- **fast-plate-ocr's 3 remaining errors** are near-misses: Q→O and D→B (visually similar characters), and a small "1/2" county marker read as an extra "2".
- **Caveat:** these 23 plates were large enough to be read by eye, so they represent *readable* plates. The 50-plate random sample in `manual_check.csv` will provide the unbiased figure.

### 7.2 OCR on the whole test set (731 detected plates)

![Text length and confidence distribution](plate_detection/report_figures/fig_ocr_testset.png)

| | EasyOCR | **fast-plate-ocr** |
|---|---|---|
| Plates with any text read | 619 (84.7%) | 624 (85.4%) |
| Plausible length (4–8 characters) | 466 (63.7%) | **602 (82.4%)** |
| Mean OCR confidence (when text read) | 0.42 | **0.92** |
| Reads with confidence ≥ 0.9 | 69 | **488** |

**Interpretation**
- **fast-plate-ocr produces plate-shaped answers far more often** (82% vs 64% with 4–8 characters). EasyOCR frequently returned fragments of 1–3 characters.
- **Its confidence is meaningful and decisive:** most reads are either very confident (≥ 0.9) or clearly not — useful for flagging uncertain reads in a user interface.
- **About 15% of detected plates yield no text** with either engine; these are mostly tiny, heavily noised, or partially visible plates (see 7.3).

### 7.3 OCR success vs. plate size

![OCR plausible-read rate by plate size](plate_detection/report_figures/fig_ocr_by_size.png)

| Detected plate height | Plates | Plausible reads | Rate | Mean OCR confidence |
|---|---|---|---|---|
| < 20 px | 14 | 1 | **7%** | 0.07 |
| 20–40 px | 27 | 16 | 59% | 0.56 |
| 40–80 px | 216 | 174 | 81% | 0.75 |
| 80–160 px | 234 | 203 | 87% | 0.82 |
| ≥ 160 px | 240 | 208 | **87%** | 0.85 |

**Interpretation**
- **Plate size limits reading even more than detection.** Below 20 px, characters are only a few pixels tall and practically unreadable for any OCR.
- **Above ~80 px the rate levels off at ~87%.** The remaining failures there are mostly mirrored, partially visible or heavily noised plates rather than size.
- **Practical rule:** for reliable reading, the camera setup should deliver plates **at least ~40–80 px tall** at 640 px resolution.

### 7.4 Before and after: the same 8 demo images

| EasyOCR | fast-plate-ocr |
|---|---|
| ![EasyOCR demo](plate_detection/ocr_outputs/ocr_demo_images_easyocr.png) | ![fast-plate-ocr demo](plate_detection/ocr_outputs/ocr_demo_images.png) |

| Plate | EasyOCR | fast-plate-ocr |
|---|---|---|
| NAB 7530 | HAB953U | ✅ NAB7530 |
| SEXEY-ME | SEXEYHEL | ✅ SEXEYME |
| GOT GOD | GOTEGOD | ✅ GOTGOD |
| HTDGKNG | HTDGKHG | ✅ HTDGKNG |
| DVJ (blurry close-up) | W | ✅ DVJ |
| Tiny plates in mosaics | ? | ? (still unreadable) |

---

## 8. End-to-End System Estimate

The share of **all real plates that are both detected and read exactly right** can be estimated as:

> **End-to-end rate ≈ detector recall × OCR plate accuracy**

| Scenario | Detector recall | OCR plate accuracy | End-to-end |
|---|---|---|---|
| Readable plates (23-crop sample) | 0.941 (conf 0.25) | 0.87 | **≈ 0.82** |
| Pending: random sample (`manual_check.csv`) | 0.941 | *to be measured* | *to be computed* |

**Interpretation:** for plates of normal size, roughly **4 out of 5 plates in an image are fully recognized** without human help. Because the 23-crop sample favours readable plates, the random-sample figure is expected to be **lower**; it should replace this estimate once `manual_check.csv` is completed.

---

## 9. Key Findings & Recommendations

### Findings
1. **Accurate detector:** test precision 0.985, F1 0.955, mAP@0.5 0.946 — trained in 4.4 h on a laptop GPU with no overfitting.
2. **Plate size is the single biggest factor** for both stages: detection recall falls from 99% (≥ 160 px) to 19% (< 20 px); OCR success falls from 87% to 7%.
3. **Threshold choice is forgiving:** F1 stays within 0.01 of its peak between confidence 0.25 and 0.7.
4. **Annotation-style inconsistency** inflates the error counts and depresses mAP@0.5:0.95.
5. **Domain-specific OCR wins decisively:** fast-plate-ocr is ~30× faster and reads 87% of readable plates exactly vs 30% for tuned EasyOCR.
6. **Real-time capable:** 65 FPS detector, 42 FPS full pipeline.

### Recommendations
| Priority | Recommendation | Expected effect |
|---|---|---|
| High | Complete the 50-plate manual OCR check | Unbiased OCR accuracy for the final report |
| High | Use confidence **0.4–0.5** for deployment | F1 ≈ 0.955; false positives roughly halved vs 0.25 |
| High | Ensure plates appear **≥ 40–80 px tall** (camera placement / zoom) | Detection ≥ 96%, plausible reads ≥ 81% |
| Medium | Re-split the dataset by source photo | Removes possible train/test overlap → more honest test scores |
| Medium | Standardize box style (whole plate) | Higher mAP@0.5:0.95; cleaner crops for OCR |
| Medium | Train and compare YOLO26s / higher resolution | Better recall on small plates |
| Low | Plate-format rules and fine-tuning OCR on local plates | Fix residual O/0, B/D/8 confusions |

---

## 10. Appendix: Figures & Reproducibility

| Figure | File |
|---|---|
| Dataset splits & plate sizes | `plate_detection/report_figures/fig_dataset.png` |
| Training losses | `plate_detection/report_figures/fig_training_losses.png` |
| Training metrics | `plate_detection/report_figures/fig_training_metrics.png` |
| Validation vs test | `plate_detection/report_figures/fig_val_vs_test.png` |
| Threshold sweep | `plate_detection/report_figures/fig_threshold_sweep.png` |
| Error breakdown | `plate_detection/report_figures/fig_error_breakdown.png` |
| Recall by size & confidence | `plate_detection/report_figures/fig_recall_by_size_and_confidence.png` |
| Speed | `plate_detection/report_figures/fig_speed.png` |
| OCR versions | `plate_detection/report_figures/fig_ocr_versions.png` |
| OCR on test set | `plate_detection/report_figures/fig_ocr_testset.png` |
| OCR by plate size | `plate_detection/report_figures/fig_ocr_by_size.png` |
| Ultralytics outputs | `plate_detection/outputs/`, `plate_detection/runs/yolo26n_640_e50_test/` |
| OCR outputs | `plate_detection/ocr_outputs/` |

**Regenerating the charts:** `python scripts/make_report_figures.py` (reads the result files in `plate_detection/` and runs the trained detector once on the test split; charts and computed numbers are written to `plate_detection/report_figures/`). The figures are not stored in the repository, so the images in this document appear after running the training, OCR and chart scripts.

**Evaluation settings used throughout:** test split (593 images, 743 plates), image size 640, IoU ≥ 0.5 for a correct detection, confidence ≥ 0.25 unless stated otherwise, NVIDIA GeForce RTX 2050 (4 GB), Ultralytics 8.4.204, fast-plate-ocr 1.1.0.
