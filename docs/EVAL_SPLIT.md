# 3DGS Train/Test Split — how it works and how to keep it comparable

The held-out **test** views are what `metrics.py` measures PSNR/SSIM/LPIPS on. For any cross-method
comparison (your pipeline vs the paper, COLMAP vs `sparse_metric` vs Agisoft) to be fair, every method
must hold out the **same physical views**. This doc explains the split and the tooling that guarantees it.

## Single source of truth

All split logic lives in [`src/wheat_utils/split_utils.py`](../src/wheat_utils/split_utils.py)
(`compute_eval_split`). Both training ([`dataset_readers.py`](../src/gaussians/scene/dataset_readers.py))
and the standalone checker import it, so a verification can never disagree with what training actually does.

Rules, in priority order:
1. **Pinned test list** → split by **name identity**. Pin sources: FIP `transforms.json` `test_filenames`
   (the paper's split), or phone `phone_split.json` `test_views`. Identity = robust to registration drift
   and to image-count differences across methods.
2. **FIP naming fallback** (every name ends `_cam_NN`): `cam_NN > 10` = test → `cam_11`, `cam_12`.
3. **llffhold-8 fallback** (phone, no pin): every 8th sorted image is test.

`--eval` is always passed by `run_reconstruction.py` (train + render), so eval mode is always on.

## FIP — already pinned to the paper

FIP plots ship `transforms.json` (NeRF format, also used by the old FruitNeRF baseline) whose
`test_filenames` are exactly `cam_11`/`cam_12` across the 3 captures (30 train / 6 test). The pipeline now
reads that file as the pin, so your FIP runs test on **the paper's exact held-out views**. Verified
byte-identical to the previous cam-index logic. FIP also never re-runs SfM (uses the shipped Agisoft
`sparse/`), so the registered set is a constant 36/36 — no drift. **If you later run FIP through your own
COLMAP**, the split stays identity-based (it can only *shrink* if a `cam_11/12` fails to register — and
`check_split.py` flags that).

## Phone — pin it once, then every method matches

Phone has no `transforms.json`, and llffhold-8 is **positional**: a different registered set shifts which
views are held out → metrics not comparable. Fix:

```bash
# write phone_split.json from the FULL intended set (input_uniform/), BEFORE/independent of COLMAP
python src/preprocessing/make_phone_split.py field=field_A plot=20250609
```

This derives the split from `input_uniform/` (every intended image, before COLMAP drops any), so it is
**method-independent**. Once `phone_split.json` sits at the session root, `dataset_readers.py` honors it for
COLMAP, `sparse_metric`, `sparse_metric_gcp`, … — all hold out the same 15 views. `overwrite=false` protects
a split you're already comparing against.

**Agisoft caveat:** Agisoft renames images (`IMG_..._<seq>`), so the COLMAP-named pin matches 0 Agisoft
names. Both training and the checker detect this (0 overlap) and **fall back** rather than empty the test set.
So Agisoft-vs-COLMAP is still not name-comparable today — a known follow-up (would need `_<seq>` normalization).

## Checking before you compare

```bash
python src/preprocessing/check_split.py dataset=fip plot=plot_461                  # FIP vs paper
python src/preprocessing/check_split.py field=field_A plot=20250609               # phone COLMAP
python src/preprocessing/check_split.py field=field_A plot=20250609 sfm_subdir=agisoft
```

Reports registered count, which inputs failed to register, the pin source, whether every pinned test view
registered, and the resulting split. Writes `logs/split_check.json`. Exit codes (usable as a gate):
- **0 = PASS** — pin satisfied (or no pin → fallback).
- **1 = FAIL** — pin applies but some test views dropped → **drift**, metrics not comparable.
- **2 = N/A** — pin exists but its naming doesn't apply here (e.g. Agisoft).

## Registration marking (#1)

`run_colmap.py` now lists the **names** of any input image it failed to register (loud `WARNING` when
< 100%), and records them as `missing_images` in `colmap_summary.json`. A missing image silently drops out
of the split, so this makes the drift visible at preprocessing time.

## Downstream consumers (what uses test vs train)

- **metrics.py** → test renders only (the headline PSNR/SSIM/LPIPS).
- **render.py** → renders both; metrics ignores the train ones.
- **render_360.py** → train cameras only (seeds the orbit).
- **viewer** → free interactive camera, split-independent.
- **eval_wheatgs.py (step 6)** → renders both splits' seg masks.
- **eval_seg_2d.py (step 6b)** → keyed on GT masks; finds the prediction in test/ *or* train/, so it's
  robust to which split a GT-labeled camera landed in.

### eval_seg_2d data availability (matters for the paper comparison)

eval_seg_2d loops over every `manual_label/*_gt_mask.png`, so its 2D-seg metric is a **mean over the
hand-labeled images** — but today the labeled set is tiny:
- FIP `plot_461`: **1** GT mask (`..._cam_12_gt_mask.png`, a test view) → the metric is effectively a
  single-image number.
- phone: **0** GT masks → eval_seg_2d skips entirely.

This is a labeling-effort limit, not a code limit. For a robust 2D-seg comparison vs the Wheat3DGS paper,
hand-label more masks (e.g. both test views `cam_11` + `cam_12` across the 3 captures) and drop the
`{camera}_gt_mask.png` files into `manual_label/` — the script picks them all up automatically and averages.
Match whatever labeled set the paper reports, same fairness principle as the train/test split itself.

## Files

- `src/wheat_utils/split_utils.py` — shared helper (the only place the rule is defined).
- `src/preprocessing/make_phone_split.py` (+ `configs/preprocessing/make_phone_split.yaml`) — writes `phone_split.json`.
- `src/preprocessing/check_split.py` (+ `configs/preprocessing/check_split.yaml`) — verify / gate.
- `src/preprocessing/run_colmap.py` — registration marking.
- `src/gaussians/scene/dataset_readers.py` — consumes the helper.
