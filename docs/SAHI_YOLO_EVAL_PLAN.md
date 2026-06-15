# SAHI vs YOLO vs GT — Evaluation/Metrics Plan (for SAHI parameter tuning)

**Status: IMPLEMENTED & smoke-tested on FIP (2026-06-15).** All four tools built and validated end-to-end on real `metrics_v2` boxes. This doc captures the design for comparing the boxes found by **SAHI** (`sahi_yolo_sam`), **plain YOLO** (`yolo_sam_v1`), and the **manual-label ground truth (GT)**, so SAHI parameters can be tuned by *seeing* what SAHI adds vs what it breaks.

**Files (built):** `src/mask_generation/evaluation/{compare_common,eval_compare_3way,eval_compare_nogt}.py`, `src/mask_generation/sahi_yolo_sam/sahi_merge_debug.py`; configs `configs/mask_generation/{eval_compare,eval_compare_nogt,sahi_merge_debug}.yaml`. Zero edits to `eval_yolo_boxes.py` (imports its primitives). Run examples:
```bash
python src/mask_generation/evaluation/eval_compare_3way.py  yolo_experiment=metrics_v2 sahi_experiment=metrics_v2 overlay_mode=both fp_singles=true
python src/mask_generation/evaluation/eval_compare_nogt.py  dataset=phone overlay_mode=both
python src/mask_generation/sahi_yolo_sam/sahi_merge_debug.py plot_glob=plot_461 limit_images=1
```
**First-run findings (metrics_v2, 7 FIP plots):** SAHI helps medium heads, *hurts* large heads at seams (R 0.942 vs YOLO 0.968); COCO small-bucket empty (heads too big → tertiles carry the analysis); SAHI over-counts +17.8% with 241 SAHI-unique seam-duplicate FPs (vs 62 YOLO-unique) and 45 splits (vs 24). YOLO↔SAHI agreement rate 0.727. → the merge / overlap is the thing to tune; use `sahi_merge_debug` + watch `count_error_ratio`.

**Goal:** tune SAHI knobs (slice size, overlap, postprocess merge) by measuring, head-by-head, what SAHI recovers vs plain YOLO and where it over-/under-merges — not just aggregate precision/recall.

**Goal:** tune SAHI knobs (slice size, overlap, postprocess merge) by measuring, head-by-head, what SAHI recovers vs plain YOLO and where it over-/under-merges — not just aggregate precision/recall.

---

## What already exists (reuse, mostly zero code)

[`src/mask_generation/evaluation/eval_yolo_boxes.py`](../src/mask_generation/evaluation/eval_yolo_boxes.py) is **method-agnostic**:
- Reads boxes from `results/mask_generation/{dataset}/{plot}/{method.name}/{experiment}/bboxes/` and matches to `manual_label/*.txt` (GT).
- `cfg.method.name` is baked into the path, and **both** `yolo_sam_v1` and `sahi_yolo_sam` write the **same** format: `bboxes/*.pt` (4-col `[x1,y1,x2,y2]`) and `bboxes_with_conf/*.pt` (5-col, all NMS-passing, for AP).
- So you can already get **method-vs-GT** numbers for SAHI just by flipping the method group:
  ```bash
  python src/mask_generation/evaluation/eval_yolo_boxes.py method=yolo_sam_v1   eval_experiment=cmp_yolo
  python src/mask_generation/evaluation/eval_yolo_boxes.py method=sahi_yolo_sam eval_experiment=cmp_sahi
  ```
- Gives P/R/F1/AP, count-error ratio, TP-IoU histograms, FP/FN heatmaps, match_viz (blue=TP, orange=FP, red=FN). **Already confirmed working on SAHI output** (ran once).

**The gap this plan fills:** `eval_yolo_boxes.py` matches each method to GT *independently*, so it never shows — head by head — **what SAHI adds vs what it breaks relative to plain YOLO.** That differential is the whole point of SAHI tuning. The new tools below import `compute_iou_matrix`, `match_boxes`, `load_gt_boxes` from `eval_yolo_boxes.py` (no changes to it).

---

## Files to build (3 new + 1 shared helper module)

| file | location | purpose | GT? |
|---|---|---|---|
| `eval_compare_3way.py` | `src/mask_generation/evaluation/` | SAHI vs YOLO vs GT — coverage table, size strata, split/merge, 7-region overlays | yes (FIP) |
| `eval_compare_nogt.py` | `src/mask_generation/evaluation/` | SAHI vs YOLO **agreement** (no GT) — divergence stats + overlay | no — **FIP + phone** |
| `sahi_merge_debug.py` | next to the SAHI method (`src/mask_generation/sahi_yolo_sam/`, imports its internals); **output** under `evaluation/` | inspect SAHI internals: per-tile + before/after-merge overlays + merge counts | n/a |
| `compare_common.py` | `src/mask_generation/evaluation/` | shared helpers for the two compare files (see below) | n/a |

**`compare_common.py`** holds the two functions both compare files need but that do NOT belong in `eval_yolo_boxes.py` (that file is GT/TP-FP-FN-centric; these are method-vs-method):
- `categorize_two_sets(A, B, iou_thr)` → mutual-match / A-only / B-only (used by nogt, and by 3way for the shared-FP cross-match).
- `draw_overlay(image, {label: (color, boxes)}, out)` → generic colored-box drawer for every overlay.

The 7-region GT Venn categorizer lives **only** in `eval_compare_3way.py` (only it has GT). All files import the matching primitives (`compute_iou_matrix`, `match_boxes`, `load_gt_boxes`, `load_pred_boxes`) from `eval_yolo_boxes.py` — **zero edits to that file.**

**The file you run IS the GT / no-GT switch** — no flag needed. `eval_compare_3way` needs `manual_label/` (FIP only); `eval_compare_nogt` runs on FIP *and* phone.

---

## Core metrics & artifacts (file 1, the tuning instrument)

### A. Per-GT-head coverage table (2×2)
Match YOLO→GT and SAHI→GT independently (greedy IoU ≥ 0.35, reuse `match_boxes`). For each GT head, check which methods' matched-set it's in:

|  | YOLO hit | YOLO miss |
|---|---|---|
| **SAHI hit** | both found | **SAHI rescued** ← the SAHI value |
| **SAHI miss** | only YOLO (regression) | neither found (hard miss) |

The "SAHI rescued" and "only YOLO" cells justify or kill a SAHI parameter set. Plus per-method FP (boxes matching no GT). Compute this table **per size bucket** (below).

### B. Stratify by GT box size (LOCKED: emit BOTH tables)
Bin each GT head by box area, report metrics/coverage **per bucket**. SAHI's reason to exist is small objects, so the gain should land in the **small** bucket; if it also lifts large heads, something's off (likely seam duplicates). **Produce two side-by-side coverage tables** (bucketing a head twice is a few extra lines):
- **Tertiles** (dataset-relative): cut at the 33rd/66th percentile of *this* dataset's GT areas → always 3 populated buckets, no magic numbers. **This is the table the tuning decisions are read off.** Print the GT area distribution first.
- **COCO fixed:** small <32²=1024 px², medium 1024–96²=9216, large >9216 — literature-comparable framing for the thesis. If FIP heads all fall in one bucket, that's itself a useful honest fact to report.

Both tables go into `compare.json` (`coverage_by_size.tertiles` + `coverage_by_size.coco`).

### C. Split / merge counting (SAHI-specific, important)
The greedy matcher mislabels two distinct SAHI failure modes; count them explicitly:
- **Split** = one GT head with **≥2** predicted boxes ≥ IoU thr (a head cut at a tile seam → two half-boxes). Shows up as 1 TP + 1 "FP", but that FP is a duplicate of a real head, **not** a hallucination.
- **Merge** = one predicted box covering **≥2** GT heads. Shows up as 1 TP + 1 "FN", but that FN is absorbed, **not** a missed head.
- **Why it matters:** splits → a *seam/overlap/merge-tuning* problem; hallucination FPs → a *YOLO confidence* problem. Plain TP/FP/FN blends them; these counts separate them so you know which knob to reach for.

### D. Visualizations — the 3-set Venn model (LOCKED)
Three box sets per image — **GT, YOLO, SAHI** (GT only on labeled FIP) — with "same box" = IoU ≥ thr. Their Venn diagram has exactly **7 regions = all possible options**, split into the recall story (GT heads) and the precision story (false boxes):

| # | region | meaning |
|---|---|---|
| 1 | GT∩YOLO∩SAHI | real head, both found it |
| 2 | GT∩YOLO\SAHI | real head, only YOLO (**SAHI regression**) |
| 3 | GT∩SAHI\YOLO | real head, only SAHI (**SAHI rescued** ← the value) |
| 4 | GT\(YOLO∪SAHI) | real head, **neither** found (hard miss) |
| 5 | YOLO∩SAHI\GT | both drew a box, no real head (**shared FP** — systematic distractor: pole/weed/shadow) |
| 6 | YOLO\(SAHI∪GT) | YOLO false box SAHI didn't draw (**YOLO-unique FP** — plain-YOLO hallucination) |
| 7 | SAHI\(YOLO∪GT) | SAHI false box YOLO didn't draw (**SAHI-unique FP** — usually a seam duplicate the merge missed) |

**Mixed themed overlays** (≤4 colors → readable, like `match_viz`):
- **Coverage overlay** = regions **1+2+3+4** (recall: what SAHI adds vs loses on real heads).
- **FP overlay** = regions **5+6+7** (precision: shared vs method-specific false boxes).

**Individual single-region images** (`overlay_mode=singles`, one color, zero clutter) for the regions where isolation adds signal — default set **{2, 3, 4, 5}**; **6 & 7** behind a `fp_singles=true` sub-toggle (region 7 vs `sahi_match_threshold`/overlap is the merge-tuning signal). **Region 1 never gets a single** — it's the bulk (~all heads), an isolated image of it shows nothing.

**NEVER one 7-region image** — 6 colors on a 500-head image is the unreadable mess we ruled out; `match_viz`-style mixing tops out at ≤4 colors, which is exactly why it's two themed overlays.

`overlay_mode` config (in BOTH compare files): `themed` (default → the mixed overlays) / `singles` (the per-region images) / `both`.

Keep the **existing per-method `match_viz`** (GT-vs-YOLO and GT-vs-SAHI TP/FP/FN) — free from `eval_yolo_boxes.py method=…`. It's now *optional* reading: Coverage (recall) + FP (precision) already separate what made the combined TP/FP/FN image busy.

### E. Cross-match FP enrichment (CONFIRMED — this is how regions 5/6/7 are computed)
Region 5 (shared-FP) comes from cross-matching YOLO-FP against SAHI-FP (IoU ≥ thr) via `categorize_two_sets`: matched pairs = shared-FP (systematic distractor), unmatched = the method-unique regions 6 & 7. Report shared-FP count in the JSON. Cheap, often revealing.

### F. Speed/cost
Report it (already measured in the mask-gen phase — just surface the number). Context: YOLO ~10–20 s, SAHI ~1–2 min, vs SAM 5–10 min and 3DGS train/seg **hours**. So SAHI's slowdown is **negligible** at the pipeline level — one line, framed as "SAHI costs ~1–2 min/image, negligible vs SAM and the multi-hour stages." Low priority.

---

## File 2: method agreement, no GT (`eval_compare_nogt.py`) — runs on FIP AND phone
No truth → can't compute TP/FP/FN. The 7 GT regions collapse to **3**: **agree** (Y∩S) / **YOLO-only** (Y\S) / **SAHI-only** (S\Y). One Agreement overlay (green=agree, blue=YOLO-only, magenta=SAHI-only) + optional singles for the two divergence regions (agree = bulk → no single). Useful signal: if SAHI-only boxes cluster in dense/small regions, slicing is doing its job. Same `categorize_two_sets` core as file 1, no TP/FP/FN labels.

**Runs on FIP too**, not just phone: `dataset=fip` → a GT-free cross-check; `dataset=phone` → the only option (no GT exists).

**Coherence with File 1 (why both on FIP is worth it):** the no-GT "agree" = with-GT regions **1+5** merged; "YOLO-only" = **2+6**; "SAHI-only" = **3+7**. GT is exactly what splits each into correct-vs-hallucinated — so running both files on FIP shows what the GT buys you.

---

## File 3: SAHI merge-debug tool (`sahi_merge_debug.py`)
The crux of SAHI tuning is the **merge** step (overlap → same head detected in multiple tiles → merge must collapse duplicates without over-merging distinct heads). Make it visible:

> **Why merge tuning is delicate (the overlap trade):** a head cut at a tile seam in one tile is *usually captured whole in the overlapping neighbor tile* (that's the point of overlap). So the merge must fuse the **whole** box with the **partial** half-boxes. Too-careful merging (high `match_threshold`) → leftover duplicate boxes (over-count); too-aggressive merging → two genuinely distinct neighbor heads get fused into one (under-count). This tool is how you find the right balance by eye + numbers.
- **Per-tile view:** tile grid on the full image + the raw boxes YOLO found in each tile (and/or saved tile crops). Spot heads cut at boundaries.
- **"Before merge" overlay:** every raw per-tile box mapped to the full image (duplicates deliberately visible — a head in an overlap region appears 2–4×).
- **"After merge" overlay:** the final boxes SAHI kept.
- **Count:** `N_raw` (pre-merge) vs `N_final` (post-merge) → `N_raw − N_final` = boxes the merge collapsed. Track vs overlap_ratio and match_threshold. (This is the *same over-counting* that `count_error_ratio` measures vs GT — the merge-debug counts show it from the inside, count_error_ratio shows the residual that survived to the final boxes.)
- **Which-merged-into-which:** color each final box's cluster (all pre-merge boxes that collapsed into it share its color). This is what lets you judge by eye: "3 duplicates of one head merged → good" vs "2 distinct heads merged → bad".

**Feasibility — CONFIRMED, the API is already factored right.** [`sahi_yolo_pipelined.py`](../src/mask_generation/sahi_yolo_sam/sahi_yolo_pipelined.py) exposes the exact four functions needed, and the **pre-merge boxes are simply `infer_tiles`'s output *before* `merge_preds` is called** (already shifted to full-image coords). The tool is **standalone** — it does NOT modify the SAHI pipeline, it imports from it and re-runs the sequence (YOLO-on-tiles only, no SAM → seconds-fast), snapshotting both stages:
```
compute_tile_boxes() → tile grid           (per-tile view)
load_and_slice()     → crops + offsets
infer_tiles()        → preds  = N_raw      ("before merge" overlay)
merge_preds()        → merged = N_final    ("after merge" overlay)
```
which-merged-into-which: match each `merged` box back to its `preds` contributors with `compute_iou_matrix` (imported from `eval_yolo_boxes`) and color each cluster. No need to disable postprocess. Run Hydra-style like the other evals; reads the SAHI knobs from `configs/.../method/sahi_yolo_sam.yaml`.

---

## Metric → SAHI-knob mapping (cheat-sheet — interpretation, NOT code)
This is a README/docstring rubric you apply while sweeping, not something to implement. The metrics it points at *are* produced by the tools above; the mapping just tells you which number reveals whether a knob helped.

| SAHI knob | watch this metric | expected direction |
|---|---|---|
| slice height/width | recall in the **small** size bucket | smaller slices → small-head recall ↑ |
| overlap_ratio | small-bucket recall **vs** SAHI-FP count | more overlap → recall ↑ but seam-duplicate FP ↑ |
| postprocess `match_metric` (IOS vs IOU) | SAHI-FP count, esp. nested boxes | **IOS** kills "small box inside big box" duplicates (the README §"Known YOLO Issues" 4(c) nested-box issue) |
| postprocess `match_threshold` | SAHI-FP count + count_error_ratio | lower → merges more aggressively → fewer dupes |

**Fastest single dial:** `count_error_ratio` (already in `eval_yolo_boxes.py`). SAHI's signature failure is **over-counting** (positive ratio) from un-merged seam duplicates — watch it while tuning postprocess.

---

## Decisions — LOCKED
- **File names:** `eval_compare_3way.py` / `eval_compare_nogt.py` / `sahi_merge_debug.py` + shared `compare_common.py`.
- **GT / no-GT switch:** the file you run (3way = GT/FIP; nogt = no-GT/FIP+phone). No flag.
- **Overlays:** 3-set Venn (7 regions); two themed mixed overlays (Coverage 1+2+3+4, FP 5+6+7) + per-region singles via `overlay_mode` (default singles set {2,3,4,5}, 6&7 behind `fp_singles`); never a 7-region image.
- **Config field names:** `yolo_method`/`sahi_method` (fixed) + `yolo_experiment`/`sahi_experiment` (the run folders).
- **Output layout** (all under the existing `evaluation/` umbrella):
  ```
  results/mask_generation/{dataset}/evaluation/
  ├── {method}/yolo_boxes/{exp}/     ← existing eval_yolo_boxes (unchanged)
  ├── compare/{exp}/                 ← eval_compare_3way (WITH GT, FIP)
  │   ├── compare.json   (2×2 coverage per size bucket, split/merge, shared-FP, speed)
  │   ├── overlay_coverage/   overlay_fp/   regions/(if singles)   config.yaml
  ├── compare_nogt/{exp}/            ← eval_compare_nogt (NO GT, FIP + phone)
  │   ├── agreement.json   overlay_agreement/   regions/(if singles)
  └── sahi_merge_debug/{exp}/{plot}/ ← sahi_merge_debug
      └── tiles/  before_merge/  after_merge/  clusters/  merge_counts.json
  ```
  (Naming/overwrite follow `eval_yolo_boxes.py`'s `get_eval_experiment` + wipe-on-run pattern.)

## Still open
*(none — all decisions locked. Size buckets: emit BOTH tertiles + COCO tables. `compare_common.py`: new module in `src/mask_generation/evaluation/`.)*

---

## Practical caveats
- **Only ~7 labeled FIP images** (one `manual_label` per plot), but each has hundreds of heads → aggregate at the **head** level, don't over-read per-image variance.
- **GT tuning is FIP-only:** phone has no manual GT yet, so the *coverage/correctness* tuning (`eval_compare_3way`) runs on FIP as a proxy. `eval_compare_nogt` still runs on phone (agreement only, no correctness). State this in the thesis.
- **Don't add elaborate statistics** — with 7 images, head-level aggregation is the honest level.
- The new tools must **not** modify `eval_yolo_boxes.py` — they import its helpers and live alongside it.

---

## Recommended build order
1. (Done) Two-arm `eval_yolo_boxes.py method=yolo_sam_v1` / `method=sahi_yolo_sam` to confirm SAHI is in the ballpark.
2. `compare_common.py` — the shared `categorize_two_sets` + `draw_overlay` (~40 lines). Build first; everything else uses it.
3. `eval_compare_3way.py` — coverage table (per size bucket) + split/merge counts + Coverage/FP overlays (+ singles, with shared-FP) + speed line. The main tuning instrument.
4. `sahi_merge_debug.py` — the merge-quality inspector (re-runs SAHI for pre-merge boxes).
5. `eval_compare_nogt.py` — method agreement (FIP cross-check, then phone once phone SAHI runs are worth comparing).
