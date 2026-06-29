# Phone mask generation: marker ROI + resolution-adaptive SAHI tiling

Two phone-only improvements to the YOLO/SAHI mask-generation stage, plus the diagnostics built to
tune them. Both are **opt-in / phone-scoped** so FIP and existing runs are byte-identical.

---

## 1. Region-of-interest (ROI) masking — `src/mask_generation/roi_mask.py`

**Problem.** Phone frames have bad corners (lens distortion, blur) and capture wheat heads from
**neighbouring plots**. Running YOLO/SAHI/SAM on the whole frame wastes effort there and produces junk.

**Idea (user-chosen).** Build a per-image ROI = the convex hull of the **6 ground markers projected
into that image**, grey out everything outside it before inference, and post-filter boxes that fall
outside. Marker polygon was chosen over square/circle because it ties the ROI to the *physical plot*.

### How it works
- We already triangulated the 6 coded markers to 3D (`logs/marker_points3d.json`, in the `sparse/0`
  frame). For each image we **project those 3D points** with the image's `sparse/0` pose + intrinsics
  (`cameras.txt` SIMPLE_PINHOLE / PINHOLE) → convex hull = plot polygon. Projecting the *3D* markers
  (not per-image 2D detections) gives a polygon in **every** image, even where a marker wasn't
  detected, and it's consistent across views. Frame note: mask-gen runs on undistorted `images/`,
  which is the same frame as `sparse/0` + `marker_points3d.json`, so the projection lands directly in
  pixel space (no undistortion of points needed).
- **Soft border (not a hard cut).** The hull joins marker *centres*, so heads on the boundary would be
  cut. The mask is the polygon **grown outward by `buffer_px`** so boundary heads stay fully visible.
- **Box post-filter.** `filter_mode="overlap"` (default) drops a box **only if it's completely outside**
  the un-buffered (true) polygon — any overlap keeps it (recall-friendly, keeps boundary heads even when
  their centre is outside). `filter_mode="center"` is the stricter centre-in-polygon variant.
- **Fallback** when `< min_markers` (3) project (e.g. FIP, failed sessions): `none` (no masking, safe
  default) / `circle` / `square`.

### Dynamic buffer (resolution-robust)
`buffer_px` was made dynamic: **`buffer_frac` × image short side** (takes precedence; `buffer_px` is the
absolute fallback). Default **`buffer_frac: 0.05`** ≈ 151 px on 4032×3024. It means "grow the polygon
boundary outward by N px in every direction" (it dilates the polygon mask). The box filter always uses
the **un-buffered** polygon. Visual tuning showed 80 px too tight, 400 px re-admits neighbour/foreground;
~120–160 px was the sweet spot, settled on 0.05 (≈151 px). NB: measured head sizes (below) later showed
the buffer should comfortably exceed the head size — at 0.05 it does.

### ⚡ Performance fix — distanceTransform, not dilate
Growing the mask with `cv2.dilate` uses a `(2·buffer+1)²` elliptical kernel → **O(buffer²)**: at
buffer 151 that's a 303×303 kernel = **~5.8 s/img** (it dominated runtime — phone YOLO went 0.5→1.45
s/img). Replaced with `cv2.distanceTransform` (distance-to-polygon ≤ buffer) = **~0.08 s/img,
buffer-independent (~70×)**. Verified equivalent: masks differ by 0.2% of pixels — a 1px boundary ring
(rasterisation rounding). Both approximate a disk dilation; same result. This made YOLO ~6× faster and
SAHI ~5× faster on the ROI step (`_roi_keep_region`).

### Wiring (all three stages) + config
- `apply_roi(img, path, cfg)` — grey-out (buffered) → called in YOLO `resize_single_image`, SAHI
  `load_and_slice`, SAM `_load_image_and_bbox`.
- `roi_keep_mask(boxes, path, cfg, w, h)` — box filter → in YOLO `save_single_result`, SAHI `_merge_and_save`.
- Config block in `configs/mask_generation/config.yaml`: `roi.enabled` (default **false** → no-op,
  byte-identical; phone runs pass `roi.enabled=true`), `source`, `fallback`, `min_markers`,
  `buffer_frac`/`buffer_px`, `filter_boxes`, `filter_mode`, `filter_tol_px`, `fill`.
- Folder reset: both detectors `reset_folder()` `bboxes/` + `yolo_vis/` per plot → a re-run **wipes**
  old output (no stale-mix), each method writes its own tree.
- Eyeball before a run: `python src/mask_generation/roi_mask.py <plot_dir> --buffer_frac 0.05` (writes
  overlays: green = true marker hull, orange = buffered edge, dark = greyed-out).

---

## 2. Resolution-adaptive SAHI tiling

**Problem 1 — wasted near-duplicate tile.** Fixed `slice=1280, overlap=0.3` on 4032×3024 → step 896 →
columns at 0/896/1792/2688 + an edge-aligned 2752 = a **64 px near-duplicate** column (15 tiles, 3 are
redundant). **Problem 2 — hardcoded pixels** break on other resolutions.

**Skipping grey tiles doesn't help** (measured 0% fully-outside the ROI at 1280 — tiles too coarse vs
the large central ROI). Tiling only the **ROI bbox** would cut ~53% of tiles, but the cleaner general
fix is dynamic sizing + even placement.

### Algorithm (`compute_tile_boxes_dynamic`, `dynamic_tile_size`)
1. Drive tile size from the **longer side** `L = max(W,H)` (handles portrait/landscape).
2. Ideal tile count `n = round((L−target)/(target·(1−overlap)) + 1)` (round, not ceil — lands nearest
   the target scale).
3. Exact even-fit size `T* = L/(1+(n−1)(1−overlap))`, then **`T = ceil(T*/32)·32 + 32`** — "best ×32
   (YOLO stride) + one 32-stride buffer" so tiles are a touch bigger than perfect-fit (a bit more
   overlap, guaranteed coverage).
4. Place tiles by **even distribution** (`_even_origins`): origins spread evenly from 0 to D−T so first
   & last sit on the edges → **no near-duplicate at any resolution**; all tiles exactly T×T.
5. The model runs at `size=T` per image (`load_and_slice` returns `infer_size`, `infer_tiles` uses it).

Verified: 4032×3024 → **1344px, 4×3=12 tiles, 0.33 overlap**; portrait 3024×4032 → 1344/12; 48 MP
8000×6000 → 1408/48; 4K 3840×2160 → 1280/8; all clean, no dup.

### Default-on for phone
`sahi_dynamic_tiles: auto` (config) → resolved by `use_dynamic_tiles(cfg)`: **ON for `dataset.name==phone`**,
OFF for FIP (keeps the fixed-1280 GT benchmark byte-identical). Explicit `true`/`false` overrides.
`sahi_target_tile: 1280`. Fixed path (`sahi_slice_size`) untouched.

### Measured comparison (field_A/20250715, 96 imgs, ROI on)
| tiling | tiles | total heads | mean/img | time | s/img |
|---|---|---|---|---|---|
| 1280 fixed (old) | 15 | 25,470 | 265.3 | 111.7s | 1.16 |
| **1344 dynamic (new default)** | 12 | 25,558 | 266.2 | **83.3s** | **0.87** |
| 2048 fixed | 6 | 25,776 | 268.5 | 95.3s | 0.99 |

Dynamic = **same count as 1280** (+0.3%, per-image mean +0.9) but **~26% faster** (drops the 3
redundant tiles), and resolution-robust. 2048 finds slightly more (+1.2%) but is slower, more VRAM, and
unvalidatable without GT. So dynamic is a strict improvement over the current default.

---

## 3. Supporting facts measured this session

- **Box sizes** (looping all boxes, 20250715): median **~73×120 px**, longer-side median ~132, p99 ~262,
  **max ~389 px**, area median ~8 k. (Earlier ~67×161 estimate was rough.) ⇒ overlap band should exceed
  ~260 px (p99): 0.3 = 384 px covers it; 0.2 = 256 px would clip the biggest ~1% — so 0.3 is justified.
- **`max_det` is already 1000**, not 300. AutoShape (`common.py:819`) overrides `general.py`'s 300, and
  the model runs through AutoShape. At ~tens of detections/tile we're nowhere near it. No change needed.
- **GWHD scale / upscaling**: GWHD images 1024², heads ~50–100 px. Phone heads native ~73×120 px. Plain
  YOLO letterboxes 4032→1280 (×0.317) → heads shrink to ~21×51 (below GWHD → why plain YOLO under-detects).
  SAHI tiles at native res → heads stay native (≈ GWHD scale) → why SAHI works. **Do not upscale** —
  SAHI already lands heads at the trained scale; bigger tiles preserve scale too (the model runs at
  `size=T` on native crops), they only change tile count/VRAM, plus fewer seams = fewer merges.
- **Confidence (kept boxes, 20250715)**: of boxes ≥0.35, only **1,114 SAHI (~4.4%)** and **1,787 YOLO
  (~10%)** are in [0.35,0.4). So raising the threshold to 0.4 costs SAHI little, YOLO more → SAHI's kept
  boxes are more confident. (A confidence histogram is valid for YOLO — NMS only suppresses; for SAHI a
  *lowered-threshold* run is NOT representative because NMM reshapes — so tune SAHI by re-running at the
  candidate threshold, or histogram only the kept-box confidences at the production threshold.)
- **Box counts YOLO vs SAHI** (20250715, thr 0.35): YOLO 17,527 (182.6/img) vs SAHI 25,470 (265.3/img),
  SAHI +82.7/img on every image — its native-res tiling recovers many more small/dense heads.

---

## 4. New flags + tooling
- `save_bboxes_conf` (config, default false) — write `bboxes_with_conf/*.pt` (5-col, with conf) for ALL
  images even without manual labels (phone has none). Decouples conf-saving from `only_labeled_images`.
- `src/analysis/compare_yolo_sahi.py` — per-image YOLO-vs-SAHI box counts (CSV + scatter/sorted plot) and
  kept-box confidence histogram. Output → `docs/analysis_results/yolo_sahi_<session>/`.
- `src/analysis/viz_sahi_tiles.py` — draw the tile grid on an image (`--dynamic`/`--slice`/`--overlap`)
  to see count + overlap by eye.

## 5. Status / caveats
- **Masks regenerated** (YOLO + SAHI, ROI on, `experiment_name=initial`) on the **12 non-lisa** phone
  sessions (glob `*/????????` excludes the 2 lisa-Pixel-6a). YOLO ~0.23 s/img (~4 min total after the
  distanceTransform fix), SAHI ~1.16 s/img. These are **YOLO-only so far** (`only_yolo=true`) — no SAM
  masks yet. lisa sessions excluded by user.
- Stale old-data phone results moved to `archive/results_phone_pre_series_v1/` (mask_generation +
  reconstruction; field_A/{20250609,20250618}, field_D/{20250523,20250530}; 20250618 confirmed old by mtime).
- **No phone GT** → all of this is count/structure-validated, not precision/recall-scored. To validate
  tile size / ROI on quality, run on FIP (which has GT). dynamic-vs-2048 quality is unresolved for that reason.
