# Mask-Generation Options (YOLO + SAM) — Method Survey

A catalogue of the detector and segmentation options considered for the
`mask_generation` stage (wheat-head bounding boxes → per-head masks), and an
honest note on **why each will or won't work for us**. The goal is one place
that records what was evaluated and the reasoning, so nothing has to be
re-researched later.

---

## 0. The current pipeline (baseline)

```
YOLOv5  (wheat_head_detection_model.pt)   →   SAM ViT-H (sam_vit_h_4b8939.pth)
  detect head bounding boxes                    box prompt → per-head binary mask
```

Two facts that shape every option below:

1. **The stage is decoupled.** YOLO emits boxes; SAM only needs *a box* — it
   doesn't care which model produced it. So the **detector** and the **mask
   model** are two independent axes you can change separately.
2. **The detector dependency is the trained weights, not the architecture.**
   `wheat_head_detection_model.pt` is believed to come from the **Global Wheat
   Challenge (GWC)** winning models, i.e. trained on the GWHD dataset
   (mostly oblique / ground-level field imagery from many countries). A `.pt`
   file encodes one architecture's exact layer shapes — **you cannot load
   YOLOv5 weights into YOLOv8/v11**; a different architecture means training a
   new model from scratch.

**Hard constraint for this project: no time/appetite to train a new detector.**
That closes the "switch to a newer/better detector" path unless a *turnkey,
pre-trained* wheat detector exists to download — and none does (see §1).
So effort goes to **training-free, inference-time** changes (§2) and to the
**mask model** (§3).

---

## 1. Detector alternatives — and why they don't help us

| Option | What it is | Turnkey weights? | Verdict |
|---|---|---|---|
| **Newer YOLO (v8/v11) on GWHD** | Ultralytics ships a `GlobalWheat2020` dataset config; `yolo train …` | ❌ you train it (~100 epochs) | **Needs training.** Also inherits GWHD's domain. Ruled out (no training). |
| **YOLOv11n-GRN** (MDPI 2025) | Improved YOLOv11 for dense/occluded heads, tiny (3.6 MB) | ❌ paper weights, no clean release | **Needs training.** Ruled out. |
| **FoMo4Wheat** (arXiv 2509.06907, 2025) | Self-supervised wheat **foundation backbone** (ViT-B/L/G, MIT) | ❌ **backbone only — no detection head** | Produces *feature embeddings*, not boxes. To detect you'd bolt on a head and train it on labeled boxes. **Ruled out** (= training). |
| **GWC challenge winners** (EfficientDet / Faster-RCNN / YOLO ensembles) | Top Kaggle/AICrowd solutions, ~91–94 mAP@0.5 on GWHD | partial (code on GitHub, heavy ensembles) | Slow, heavy integration, still domain-bound. Not worth it. |
| **SAM 3 open-vocabulary as the detector** | Prompt `"wheat head"` (text) or one exemplar box → finds the rest, training-free | ✅ (it's a released model) | **Untested for us.** Training-free is attractive, but open-vocab models typically *underperform* a specialized detector on extremely dense, small, overlapping heads. Worth a *quick qualitative* try on phone (different failure mode), not a reliable replacement. |

**Bottom line:** there is **no turnkey wheat detector better than what we have**
that doesn't require training. The detector axis is effectively frozen on the
existing GWC YOLOv5 weights.

---

## 2. Inference-time improvements (training-free — the actionable lever)

These wrap the **existing** YOLOv5 weights, no retraining.

### 2a. SAHI — Slicing Aided Hyper Inference  ⭐ top pick for phone
*(arXiv:2202.06934)*

- **What:** tiles each image into overlapping crops (e.g. 640², overlap ~0.2),
  runs the detector on each tile, then merges boxes (NMS / NMM).
- **Why it fits our phone problem:** phone images are high-res with **many
  small, dense, overlapping heads**. Today the whole image is letterboxed down
  to 1280 in one pass, so heads shrink and get missed. SAHI keeps **native
  resolution inside each tile**, so each head is large relative to its tile →
  recall jumps, especially on overlapping clusters.
- **Relation to the current `target_image_size: 1280`:** bumping 640→1280 was
  a crude, fixed-size version of the same idea — but a 4000 px image at 1280 is
  still ~3× downscaled. SAHI is the proper, *adaptive* version (tile-count
  scales with image size) and strictly preserves more detail.
- **Costs / knobs:** slower (N tiles per image); tune slice size, overlap
  ratio, and the merge step (NMS vs NMM for overlapping heads at tile seams).
- **Integration note:** we use a **vendored** yolov5 (`src/mask_generation/yolov5/`),
  not the pip package, so SAHI needs a small adapter to its detection-model
  API rather than the one-line path. Feasible, not free.
- **Verdict:** highest-value, lowest-risk move for phone. Targets the actual
  cause (small/dense) and reuses the existing weights.

### 2b. Raise `target_image_size` (already done: 640 → 1280)
- The crude version of SAHI. Helps small objects (more pixels per head) but is
  a single global downscale and hits VRAM/compute limits. SAHI supersedes it.

---

## 3. Mask-model (SAM) alternatives — the second, independent axis

The detector stays the same; only the box→mask model changes. All take box
prompts, so they're drop-in. Relevant because mask-edge quality compounds
across thousands of densely packed heads (see `MASK_SIZE_ANALYSIS.md`).

| Option | What it adds | Fit |
|---|---|---|
| **SAM 2** | faster image+video encoder, better masks than ViT-H | drop-in mask-quality upgrade, training-free |
| **SAM 3** (2025) | open-vocab text + exemplar prompts; stronger encoder; still accepts box prompts | drop-in for masks **and** an optional open-vocab detector experiment (§1) |
| **HQ-SAM** | high-quality mask head for fine/thin structures (awns, edges) | promising for fine wheat detail; small add-on |
| **MobileSAM** | tiny/fast SAM | speed play if SAM is the bottleneck; some quality trade-off |

These are all **mask-quality** changes — they do **not** fix missed
detections (that's the detector/SAHI axis).

---

## 4. Phone-specific diagnosis (why phone looks worse than FIP)

FIP detection works well; phone "looks kinda bad." The differences, ranked by
how likely they're the cause:

| Difference | FIP | Phone | Cause? | Fix |
|---|---|---|---|---|
| **Head density / scale** | fewer, larger heads | **many small, overlapping heads** | **likely main cause** | **SAHI** (§2a) |
| **Image format** | **PNG (lossless)** | **JPG (lossy, camera-baked)** | plausible contributor — fine awn/edge detail destroyed before YOLO sees it | none (re-shoot lossless); SAHI stops *adding* resolution loss on top |
| **Viewpoint** | overhead | side / oblique | **probably NOT the cause** — GWC weights were trained on oblique imagery, so phone is *closer* to the training domain than FIP is | n/a |

Two things to internalize:
- The **downscale to 1280 is not a JPEG step** — it's an in-memory LANCZOS
  resize, no re-encoding. The lossy JPEG already happened in the phone camera;
  resizing can't recover it. FIP PNGs carry zero JPEG.
- Because the weights are **GWC-trained (oblique)**, the side-view is not the
  problem — which is *good news*: it means the fixable axes (density via SAHI,
  and capture-time JPEG) are where the gains are.

---

## 5. Recommendation / priority

1. **SAHI** around the existing YOLOv5 weights — directly targets the dense /
   small-head phone problem, training-free. (§2a)
2. **SAM 3 / HQ-SAM** as a mask-quality upgrade (and a quick open-vocab SAM 3
   detection experiment on phone for curiosity). (§3, §1)
3. Capture **lossless (PNG)** phone images when re-shooting, to remove the
   JPEG variable entirely. (§4)

Detector replacement is **off the table** (no turnkey option without training).

---

## 6. TODO / open items

- [ ] **Phone ground-truth labels.** We currently have **no GT masks/boxes for
  phone**, so nothing on phone can be *measured* — only eyeballed. Need to ask
  the supervisor how their manual labeling was done (tool + protocol) and
  reproduce it for a small representative phone set. Until then,
  `eval_yolo_boxes.py` can't score phone, so any SAHI / SAM upgrade can only be
  judged qualitatively. **Blocked on supervisor's labeling method.**
- [ ] SAHI adapter for the vendored yolov5 (§2a integration note).
- [ ] Quick qualitative SAM 3 open-vocab (`"wheat head"`) run on one phone image.

---

*References:*
*SAHI — arXiv:2202.06934 · FoMo4Wheat — arXiv:2509.06907 · YOLOv11 wheat — MDPI Agriculture 15(16):1765 · GWHD/GWC — Plant Phenomics 2020/3521852 & PMC10795497 · SAM 3 — Ultralytics docs / Meta 2025.*
