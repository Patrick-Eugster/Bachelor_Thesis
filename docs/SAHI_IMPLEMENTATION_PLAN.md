# SAHI Implementation Plan

The design + as-built record for adding **SAHI** (tiled inference) as a new
mask-generation method, without touching the working `yolo_sam_v1` pipeline.

> ## ✅ STATUS (2026-06-11): IMPLEMENTED — detection (YOLO) verified to RUN on FIP plot_461.
> Phases 1 + 2 + the Hydra config fix are done. The detector half (tiling →
> batched YOLO → NMM merge → `bboxes/` + `yolo_vis/`) **runs**; SAM half is the
> shared unchanged `sam_v1/` phase. Full quality/metrics comparison vs `yolo_sam_v1`
> still TODO (and phone needs GT). As-built details: §14; how to run: §15.

- *Why SAHI / what problem it solves:* `SAHI_EXPLAINED.md`
- *Where SAHI sits among detector/SAM options:* `MASK_GENERATION_OPTIONS.md`

---

## 1. Goal & scope

Run our **existing GWC-trained YOLOv5 weights** on **tiles** of each image
(native resolution, no downscale) so the small/dense/overlapping heads in
**phone** images are detected, instead of being lost when the whole frame is
letterboxed down to 1280. Training-free; reuses the current weights.

**In scope:** a new detector method `sahi_yolo_sam`. **Out of scope:** any change
to SAM, metrics, or the reconstruction pipeline (beyond one opt-in config field).

---

## 2. The contract to preserve

The detection→SAM handshake is on disk: per image, a `.pt` tensor of boxes
`[x1,y1,x2,y2]` in **original-image coordinates** in `bboxes/`, plus an optional
5-col `[…,conf]` in `bboxes_with_conf/` for metrics. SAM reads `bboxes/*.pt`.

**SAHI must emit the exact same files in the same format.** Then SAM, metrics,
and seg are unaffected — SAHI only changes *how the boxes are produced*.

---

## 3. Target folder structure

```
src/mask_generation/
    run_mask_generation.py     ← NEW: thin orchestrator, single entry point
    sam_v1/                    ← NEW: shared SAM phase (EXTRACTED from yolo_sam_v1, not copied)
                                  holds sam_v1_pipelined.py — "_v1" leaves room for sam_v2/ (SAM2/SAM3)
    yolo_sam_v1/               ← UNCHANGED detector + main_v1.py (Option A: keep the name)
    sahi_yolo_sam/             ← NEW: SAHI detector only (no "v1" in the name)
    metrics/  weights/  yolov5/ ← untouched
```

- **SAM is extracted, not duplicated** — one source of truth, both detectors
  import it symmetrically, no drift risk. SAM's code does not change, only its
  location. **Module name `sam_v1/`** (not bare `sam/`) so a future SAM2/SAM3
  becomes a sibling `sam_v2/` without renaming this one.
- **Option A naming:** `yolo_sam_v1/` keeps its name even though SAM moved out —
  the folder represents the YOLO+SAM v1 *method* (its `main_v1.py` still wires
  YOLO→SAM), and "yolo_sam_v1" stays the method name + result-folder string, so
  all three uses stay aligned and existing FIP results are untouched.
- `yolo_sam_v1/main_v1.py` stays a valid entry point with its sam-import
  repointed to `sam_v1/`.

---

## 4. The orchestrator + dispatch

`run_mask_generation.py` is a **thin** `@hydra.main` (NOT a copy of
`run_reconstruction.py`'s subprocess/dependency-graph/RunContext machinery —
that exists for 7+ heavy independent steps; mask-gen is just detect→segment,
in-process, sharing one model lifecycle).

"Dispatch" = pick which detector to run, by method name, at runtime:

```python
DETECTORS = {                          # name → detector function
    "yolo_sam_v1":   run_yolo_phase,
    "sahi_yolo_sam": run_yolo_phase_sahi,
}

def main(cfg):
    run_detector = DETECTORS[cfg.method.name]   # dispatch
    run_detector(image_folders, cfg)
    if not cfg.method.only_yolo:
        run_sam_phase(image_folders, cfg)       # shared SAM, always
    # report
```

This finally makes the Hydra `method=` switch change **code**, not just params.
Adding a future method (e.g. a SAM2 variant) = one import + one dict line.
Each method yaml gains a `name:` field so `cfg.method.name` resolves.

`run.py` line 20 re-points from `yolo_sam_v1/main_v1.py` to
`run_mask_generation.py`.

---

## 5. SAHI detector internals (`sahi_yolo_sam/sahi_yolo_pipelined.py`)

Per chunk of images:

```
1. SLICE    each image → native slice² tiles + each tile's (offset_x, offset_y).
            slice = imgsz (no scaling). Edge tiles padded to slice² with grey(114)
            so they stay batchable.
2. INFER    run YOLO on tiles in sub-batches (reuse the existing batched loop) →
            boxes in TILE-local coords.
3. UNSHIFT  box_global = box_tile + (offset_x, offset_y).   (no /r — native res)
4. MERGE    per image: combine all tiles' boxes with NMM + IOS match metric
            (dedupes the duplicates that overlap creates; stitches seam fragments).
5. SPLIT+SAVE  apply conf_threshold_detection (good/bad), draw, save bboxes/*.pt +
            bboxes_with_conf/*.pt + yolo_vis/*.jpg  — identical to the current
            save_single_result, just fed the merged boxes.
```

- **No scaling (detail preserved):** slice size = `target_image_size` so tiles
  enter YOLO 1:1. Recommended `slice=imgsz=1280` → ~12 tiles on a 3850×2928
  phone frame (see `SAHI_EXPLAINED.md` §3).
- **Merge** is the only non-trivial part. Use the `sahi` pip package for *just*
  its `slice_image()` + `NMMPostprocess` utilities (tested), **not** its
  sequential `get_sliced_prediction()` loop (that would throw away our
  batching). Add `sahi` to `pyproject.toml`; `pip install -e .` on Euler.
- **Optional full-image pass** (`sahi_full_image_pass`): also run one normal
  whole-image pass and merge it in — backstops heads larger than the overlap
  band (`SAHI_EXPLAINED.md` §6).

---

## 6. Config additions

**`configs/mask_generation/method/sahi_yolo_sam.yaml`** (new method, inherits the
yolo_sam params + adds):
```yaml
name: sahi_yolo_sam          # used by the dispatch registry
sahi_slice_size: 1280        # = target_image_size → no scaling, native detail
sahi_overlap_ratio: 0.3      # covers the ~161px elongated heads
sahi_merge: "NMM"            # NMM (merge) > NMS (drop) at seams
sahi_match_metric: "IOS"     # intersection-over-smaller — robust to fragments
sahi_match_threshold: 0.5
sahi_full_image_pass: true   # whole-image safety net for big heads
sahi_tile_batch_size: 8      # tiles per GPU pass (VRAM cap; 1280² ≈ 4× a 640²)
```
The existing `method/yolo_sam_v1.yaml` gains `name: yolo_sam_v1`.

**Result-folder = Option C (separate per-method trees).** The folder name
`"yolo_sam_v1"` is hardcoded in **three** consumers that must all agree:

- **Writer (`path_utils.py:54`):** `"yolo_sam_v1"` → `cfg.method.name`.
- **Seg reader (`path_utils.py:78`):** → `cfg.segmentation_3d.detection_method`,
  a **new field** in `configs/reconstruction_seg3d/segmentation_3d/default.yaml`
  (default `yolo_sam_v1`).
- **Metrics reader (`metrics_yolo_v1.py:765`, eval-out dir `:805`):** →
  `cfg.method.name` — `metrics_eval.yaml` already loads the method group so it
  resolves. (Found via grep; same one-line pattern as the others.)

These live in different config trees (mask-gen vs reconstruction), so they're
separate fields kept in sync, not one shared value. **Backward-compatible:** the current
method is already named `yolo_sam_v1`, so existing write/read paths are byte-for-
byte unchanged; only a SAHI run (`method=sahi_yolo_sam` +
`segmentation_3d.detection_method=sahi_yolo_sam`) lands in a new tree.

---

## 7. Parallelism & memory

- **Sequential vs batched:** vanilla SAHI processes tiles one-at-a-time. We don't
  use that loop — tiles flow through the **existing batched** sub-batch
  mechanism, so the GPU stays saturated.
- **VRAM:** peak = `sahi_tile_batch_size × one slice² forward`. A single 1280²
  tile ≈ the same VRAM as today's single 1280 whole-image pass. Sequential
  (`tile_batch=1`) costs the same VRAM as now; batching trades VRAM for speed —
  cap `sahi_tile_batch_size` to fit 16 GB (start 8).
- **RAM:** holds the full image + its overlapping tiles (~1.4× image pixels) —
  negligible on the 35 GB WSL budget.

---

## 8. Validation plan

1. **Regression:** `method=yolo_sam_v1` output identical to a current run
   (orchestrator + SAM-extraction must be behavior-preserving).
2. **FIP plot_461 (has GT):** `method=sahi_yolo_sam`, then `metrics_yolo_v1.py`
   vs baseline. FIP heads are large/sparse so SAHI may not *help* there — the
   point is to confirm the **merge doesn't double-count** at seams (AP must not
   drop). Cheapest correctness test available.
3. **Phone:** qualitative via `yolo_vis/` only — **no phone GT yet** (blocked on
   the supervisor's labeling method; see `MASK_GENERATION_OPTIONS.md` §6).
4. **VRAM probe:** confirm peak fits 16 GB at the chosen `sahi_tile_batch_size`.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Merge over-fuses two real adjacent heads (wheat is dense) | start IOS + threshold 0.5, **inspect `yolo_vis/`**, tune; #1 thing to watch |
| Grey-padded edge tiles spawn phantom boxes | clip boxes to real image extent; drop boxes lying in the pad region |
| "Total heads" count double-counts pre-merge | compute printed totals **after** merge |
| ~12× slower | acceptable (offline, batched); tune `sahi_tile_batch_size` |
| Writer/reader folder fields drift out of sync | both default to `yolo_sam_v1`; document the pairing |

---

## 10. What gets touched / back-compat

| File | Change |
|---|---|
| `src/mask_generation/run_mask_generation.py` | **new** thin orchestrator + dispatch |
| `src/mask_generation/sam_v1/…` | **new** — SAM moved here (no code change) |
| `src/mask_generation/yolo_sam_v1/main_v1.py` | 1-line sam-import repoint (still works) |
| `src/mask_generation/sahi_yolo_sam/…` | **new** SAHI detector |
| `configs/mask_generation/method/sahi_yolo_sam.yaml` | **new** |
| `configs/mask_generation/method/yolo_sam_v1.yaml` | add `name:` field |
| `configs/.../segmentation_3d/default.yaml` | add `detection_method:` (default `yolo_sam_v1`) |
| `src/wheat_utils/path_utils.py` | lines 54 & 78 → config-driven (Option C) |
| `src/mask_generation/metrics/metrics_yolo_v1.py` | lines 765 & 805 → `cfg.method.name` (3rd Option-C consumer) |
| `run.py` | re-point to `run_mask_generation.py` |
| `pyproject.toml` | add `sahi` |

---

## 14. As built (2026-06-11) — two deviations from the plan above

Implemented and verified (compile + Hydra config composition; GPU detection run is the user's to do). Two things differ from the plan:

1. **SAHI method config is self-contained, NOT a Hydra defaults-inherit of `yolo_sam_v1`.**
   §6's `defaults: [yolo_sam_v1, _self_]` idea does not compose under Hydra 1.3 here
   (a config-group member that redefines its own group can't be cleanly overridden —
   it collapsed `cfg.method` to the string `"sahi_yolo_sam"`). So
   `configs/method/sahi_yolo_sam.yaml` duplicates the YOLO+SAM params + adds the
   `sahi_*` block. Upside: each method now tunes its thresholds independently.

2. **`method` config group stays NESTED at `configs/mask_generation/method/`**, and
   the clean `method=sahi_yolo_sam` override is made to work via a **Hydra
   SearchPathPlugin** — no files moved. (A brief move to top-level `configs/method/`
   was tried and reverted — inconsistent with how `reconstruction_seg3d/` keeps its
   groups nested.)
   - **Why the override was broken:** a Hydra group's name is its folder path from
     the config root. With the root at `configs/`, `method/` (at
     `configs/mask_generation/method/`) was named `mask_generation/method`, so
     `method=...` matched no group and Hydra set a scalar string. (Repo-wide latent
     issue — reconstruction's `reconstruction=2dgs` / `segmentation_3d=tight`
     comments were equally broken and never exercised.)
   - **The fix:** each mask_generation entry now roots its Hydra `config_path`
     INSIDE its own folder (`config_path=…/configs/mask_generation`,
     `config_name=config`), so its `method/` group is named just `method` → clean
     override. The shared top-level `configs/dataset/` group would then be out of
     reach, so **`src/hydra_plugins/wheat_searchpath/`** adds `configs/` back to the
     search path. The plugin derives `configs/` from its own `__file__` (no
     hardcoded path) → **verified portable: works from any checkout location, Euler
     included.** Hydra auto-discovers it because `hydra_plugins` is a namespace
     package and `src/` is on `sys.path` via the editable install (flat `.pth`);
     `pip install -e .` re-creates that on a fresh machine.

**How to run SAHI:**
```bash
python src/mask_generation/run_mask_generation.py method=sahi_yolo_sam experiment_name=sahi_test
# results → results/mask_generation/<dataset>/<plot>/sahi_yolo_sam/sahi_test/   (Option C separate tree)
# to segment that SAHI output later: run_reconstruction ... segmentation_3d.detection_method=sahi_yolo_sam
```

3. **`--config-name` for the metrics configs changed** (consequence of #2):
   `--config-name mask_generation/metrics` → `--config-name metrics` (the config
   root is now inside `configs/mask_generation/`). Updated in the metrics docstring,
   both module READMEs, and `metrics_eval.yaml`.

*(Optional follow-up for repo-wide consistency: apply the same one-line
`config_path` change to the reconstruction entry so `reconstruction=2dgs` /
`segmentation_3d=tight` also work — the plugin is already global.)*

---

## 13. Locked decisions (confirmed 2026-06-11)

- **Naming:** Option A — `yolo_sam_v1/` keeps its name; SAM extracted to **`sam_v1/`** (versioned, room for `sam_v2/`).
- **Merge:** use the **`sahi` pip package** (`slice_image` + `NMMPostprocess`); hand-roll only later if needed.
- **First version:** **simple correctness-first loop** (slice → batched infer → merge → save); optimize the 3-stage pipelining later.
- **Validation:** correctness on FIP now (merge must not double-count → AP must not drop); the real phone-recall payoff is measured **later** once phone GT exists (metrics folder is ready for it).
- **Option C** spans **three** path consumers (writer + seg reader + metrics reader), all `cfg.method.name`-driven, all defaulting to `yolo_sam_v1` → existing FIP runs byte-identical.

Existing FIP results, paths, SAM, and metrics behavior are **unchanged** —
everything keys off the current method name `yolo_sam_v1`.

---

## 11. Open dependencies / blockers

- `sahi` pip package (for `slice_image` + `NMMPostprocess`) → `pip install -e .`
  on Euler too.
- **Phone ground truth** — until it exists, SAHI's phone gain is qualitative
  only. Blocked on the supervisor's labeling method.

---

## 12. Effort estimate

One ~250-line detector + a thin orchestrator + SAM extraction (move + 1 import) +
~8 config lines + 2 path_utils edits + `pyproject`. The existing detector, SAM
logic, metrics, and reconstruction are otherwise untouched. ~half a day including
merge-threshold tuning.

---

## 15. How to run + key behaviours (as-built reference)

**Run (clean `method=` override works via the searchpath plugin, §14):**
```bash
# baseline vs SAHI on ONE FIP plot, detection-only (skip SAM), same exp name (Option C → separate trees)
python src/mask_generation/run_mask_generation.py \
  dataset.plot_glob=plot_461 method.only_yolo=true experiment_name=test461
python src/mask_generation/run_mask_generation.py method=sahi_yolo_sam \
  dataset.plot_glob=plot_461 method.only_yolo=true experiment_name=test461
# compare box counts:
python src/analysis/compare_bboxes.py \
  results/mask_generation/fip/plot_461/yolo_sam_v1/test461/bboxes \
  results/mask_generation/fip/plot_461/sahi_yolo_sam/test461/bboxes
```
- **`dataset.plot_glob=plot_461` is REQUIRED to target one plot** — the default
  (`plot_glob: "*"`, `limit_plots: 0`) processes ALL 7 FIP plots (461–467).
  (`limit_plots=1` also works since 461 sorts first.)
- `method.only_yolo=true` skips SAM (the two detectors differ only in the boxes).
- FIP plot_461 has GT for only **1 image** (`cam_12`) → quantitative metrics are
  thin; the practical comparison is visual (`yolo_vis/`) + `compare_bboxes`.

**FIP geometry sanity (verified):** FIP images are **4095×2996**, so SAHI tiles
them into **15 native 1280×1280 tiles** — tiling IS exercised on FIP (it's a real
merge-correctness test, even though SAHI's real payoff is the dense *phone* heads).

**Two behaviours worth remembering:**
1. **`yolo_vis` is a whole-image overlay, not per-tile.** Each tile's boxes are
   shifted to original-image coords *during detection* (add tile offset), merged
   (NMM), then drawn on the full original image → one overlay per image, identical
   in format to `yolo_sam_v1`. Tiling is invisible in the output.
2. **Image size need NOT be divisible by `slice_size`; nothing is resized.**
   `compute_tile_boxes` steps by `slice×(1−overlap)` but shifts the *last* tile per
   row/col to end exactly at the image edge (`x0 = W − slice`). So every tile is
   full native `slice_size` (no letterbox/resize → fine detail preserved); the
   non-divisible remainder is absorbed as *extra overlap* on the edge tiles, which
   the merge dedups. (Only an image SMALLER than `slice_size` gets AutoShape-
   letterboxed — doesn't apply to FIP/phone, both > 1280.)
