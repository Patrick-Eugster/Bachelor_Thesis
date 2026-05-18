# Phone Image Preprocessing — `src/preprocessing/`

## Overview

Phone images need two cleanup steps before they can be used by the 3DGS pipeline:

1. **`preprocess_uniform_size.py`** — center-crops all images to a single resolution. Fixes the HDR-mode size mismatch that splits COLMAP's reconstruction into disconnected sub-models. **Run this first.**
2. **`convert.py`** — runs COLMAP Structure-from-Motion: feature extraction → matching → SfM mapper → image undistortion. Produces `images/` + `sparse/0/` ready for 3DGS training.

FIP plots come pre-calibrated (Agisoft Metashape, stored in COLMAP format) so neither step is needed there. Phone data goes through both.

Both scripts are configured via Hydra. Configs live in `configs/preprocessing/`.

---

## Step 1 — `preprocess_uniform_size.py` (center-crop to one resolution)

### Why this step exists

Many phones output images with **slightly different resolutions** within the same capture session. For example, our test data had:
- 66 images at 3850×2928 (one pipeline branch)
- 27 images at 3852×2936 (another pipeline branch)

These differences are tiny (a few pixels at the borders), but **COLMAP treats them as separate cameras**, which causes two problems:
- If you pass `--single_camera 1`, COLMAP refuses to start ("Single camera specified, but images have different dimensions")
- If you don't pass it, COLMAP creates separate intrinsic groups per resolution. The mapper may fail to bridge them and split into multiple disconnected sub-models (e.g. one sub-model with 29 images, another with 66), of which 3DGS can only use the largest

**The fix:** center-crop all images down to the majority resolution. We crop rather than resize so the focal length and optical center stay identical — we only trim border pixels, no real information lost. After this step every image is exactly the same size and COLMAP can treat them as a single camera.

### Why phones produce mixed sizes in the first place

We initially assumed "HDR auto-switching" was the cause, but the real reasons are usually one of these (often more than one in combination):

1. **Computational photography pipelines branching** — modern phones (Pixel, iPhone, recent Samsung, etc.) have separate ISP/software pipelines for different conditions: HDR, Night mode, Motion/Action mode, Portrait mode, standard "shot to shot." When the camera detects tricky lighting or motion mid-capture, it silently switches pipeline. Each pipeline has its own multi-frame merging step that trims a slightly different region. The 27 "different" shots in our test are probably the ones where a different pipeline got triggered.

2. **Electronic image stabilization (EIS) crop variation** — EIS works by capturing a slightly larger sensor region than needed, then cropping out a "stable window" that compensates for hand shake. When the phone detects more motion, it crops more aggressively to give EIS more room. Different motion levels per shot → slightly different output dimensions. A roughly 1:3 ratio (like our 27:66) is plausible for "shots where I moved more vs less while walking the row."

3. **Multi-frame HDR merging** — HDR shots typically capture 3–9 frames at different exposures and align + merge them. The alignment step trims edges where frames don't all cover the same pixels. Subject content + camera motion influence how much gets trimmed. Different shots → different trim amounts → different output sizes.

4. **Sensor binning / readout mode switching** — some phones use the full sensor resolution in good light but switch to a 4:1 binned mode in low light (4× more sensitive, with a slightly different effective sensor area). Less common but possible.

Less likely but still possible:
- Accidentally toggled a setting mid-capture (zoom level, aspect ratio)
- The phone overheated and dropped to a different capture mode
- Sensor temperature compensation cropped extra calibration pixels

**Looking at our specific numbers:**
- 3850×2928 (66 images) vs 3852×2936 (27 images)
- Difference: **2 pixels wider, 8 pixels taller** in the smaller group
- Both have ~11.3 MP — essentially the same

That tiny 2/8 pixel difference is way too small to be a deliberate user setting change. It's almost certainly something the ISP pipeline did silently — pipeline branch X trims 2 px more from each side than pipeline branch Y. Likely tied to which alignment / merge path was used internally for those specific shots.

**How to find out for sure (per-dataset):** check the EXIF metadata of one image from each size group:

```bash
exiftool input_plots/phone/field_A/<plot>/input/IMG_XXX.jpg | grep -iE "scene|mode|exposure|hdr|stabili"
```

If the EXIF tags differ between the two groups (e.g. "Scene Type", "HDR Mode", "Shot Mode"), you'll know which pipeline split happened. Knowing the root cause is mostly academic — `preprocess_uniform_size.py` fixes the symptom regardless of which of these caused it.

**Practical advice when capturing future datasets:**
- Lock the camera into a single mode if possible (disable Auto-HDR, lock exposure)
- Avoid mixing photo and video bursts in the same dataset
- Keep a steady walking pace so EIS settings stay consistent
- After capture, run `preprocess_uniform_size.py` *defensively* even if you think the sizes match — it's free if all images are already uniform

### How to run

```bash
python src/preprocessing/preprocess_uniform_size.py plot=20250618
```

Default paths assume `dataset=phone` and `field=field_A` — the script reads from `input_plots/phone/field_A/${plot}/input/` and writes the cropped output to `input_plots/phone/field_A/${plot}/input_uniform/` (the originals stay untouched).

### CLI options (override on command line)

| Key | Default | Meaning |
|-----|---------|---------|
| `plot` | required | Plot leaf folder name (e.g. `20250618`, `colmap_test_clean`) |
| `field` | `field_A` | Field subfolder under `${dataset.input_dir}/` |
| `dataset` | `phone` | Hydra config group; pulls `input_dir` etc. from `configs/dataset/phone.yaml` |
| `source` | auto `${input_dir}/${field}/${plot}/input` | Direct override of the source folder |
| `output` | `"" → <source>_uniform/` | Direct override of the output folder |

Defaults are stored in [`configs/preprocessing/uniform_size.yaml`](../../configs/preprocessing/uniform_size.yaml).

---

## Step 2 — `convert.py` (COLMAP Structure-from-Motion)

### What it does

Wraps four COLMAP commands into one Python script:

| Step | COLMAP command | What it does |
|------|----------------|--------------|
| 1 | `feature_extractor` | Detects SIFT features in every image |
| 2 | `sequential_matcher` or `exhaustive_matcher` | Finds matching features across image pairs |
| 3 | `mapper` | Runs Structure-from-Motion (SfM) — recovers camera positions + sparse 3D points |
| 4 | `image_undistorter` | Removes lens distortion → final `images/` + `sparse/0/` ready for 3DGS |

Output layout matches what 3DGS expects:
```
{source_path}/
├── input/                    ← your raw images (placed here before running)
├── images/                   ← undistorted images, used by 3DGS training
├── sparse/0/                 ← camera poses + 3D points (cameras.bin, images.bin, points3D.bin)
├── distorted/                ← intermediate working files (can be deleted after)
├── stereo/                   ← created by undistorter, not used by our pipeline
└── logs/
    ├── colmap.log            ← full COLMAP output, saved automatically
    └── convert_config.yaml   ← snapshot of the config used for this run
```

### How to run

After running `preprocess_uniform_size.py` (step 1), just:

```bash
python src/preprocessing/convert.py plot=20250618
```

Default paths assume `dataset=phone` and `field=field_A`. `source_path` is auto-derived as `input_plots/phone/field_A/${plot}` and COLMAP reads images from `${source_path}/input_uniform/` (the output of step 1). Originals in `input/` stay untouched.

After it finishes, the data is ready for 3DGS:
```bash
python src/run_reconstruction.py dataset=phone plot=field_A date=20250618 run_train=true
```

### CLI options (override on command line)

All options have defaults so you can usually run with just `plot=...`. Defaults live in [`configs/preprocessing/convert.yaml`](../../configs/preprocessing/convert.yaml).

| Key | Default | Meaning |
|-----|---------|---------|
| `plot` | required | Plot leaf folder name (e.g. `20250618`, `colmap_test_clean`) |
| `field` | `field_A` | Field subfolder under `${dataset.input_dir}/` |
| `dataset` | `phone` | Hydra config group |
| `source_path` | auto `${input_dir}/${field}/${plot}` | Direct override of the dataset folder |
| `image_subdir` | `input_uniform` | Subfolder COLMAP reads images from. Pass `image_subdir=input` to skip the uniform-size step |
| `camera` | `SIMPLE_PINHOLE` | Camera model — see table + test results below. Default chosen empirically: only model that worked on phone wheat data |
| `matcher` | `exhaustive` | `exhaustive` (all-pairs — robust, ~10s extra on GPU for 100 images) or `sequential` (ordered walk, low RAM, switch for 500+ images) |
| `sequential_overlap` | `25` | Sequential only: how many next images each image matches against (ignored when `matcher=exhaustive`) |
| `single_camera` | `true` | Force all images to share one camera (1 set of intrinsics solved using every image). Requires uniform dimensions. Set `false` if mixing phones/lenses |
| `num_threads` | `8` | Threads for SIFT extraction + matching — **lower = less RAM**. See RAM table below. Set `-1` for all cores |
| `no_gpu` | `false` | Set `true` to force CPU SIFT (our COLMAP build has CUDA — leave false for GPU speedup) |
| `skip_matching` | `false` | Skip steps 1–3 if you already have a working `distorted/sparse/0/` |
| `resize` | `false` | Also create downscaled `images_2/`, `images_4/`, `images_8/` folders (needs ImageMagick) |
| `colmap_executable` | `""` | Explicit binary path; empty = use `colmap` on PATH |
| `magick_executable` | `""` | Explicit binary path; empty = use `magick` on PATH |

Example overrides:
```bash
python src/preprocessing/convert.py plot=20250618 camera=PINHOLE                          # different camera model
python src/preprocessing/convert.py plot=20250618 num_threads=4 no_gpu=true               # CPU fallback
python src/preprocessing/convert.py plot=20250618 image_subdir=input                      # use original input/ folder
python src/preprocessing/convert.py plot=20250618 field=field_B                           # different field
python src/preprocessing/convert.py plot=big_dataset matcher=sequential sequential_overlap=40  # large dataset
python src/preprocessing/convert.py plot=mixed_cams single_camera=false                   # mixed phones/lenses
```

---

## CPU vs GPU SIFT

The COLMAP we built has CUDA enabled (`COLMAP 4.1.0 with CUDA` in the help output). **Building CUDA-enabled COLMAP from source is documented step-by-step in [`docs/INSTALL_COLMAP_CUDA.md`](../../docs/INSTALL_COLMAP_CUDA.md)** — use that if you need to reinstall in a fresh container.

GPU SIFT runs about **5–8× faster than CPU SIFT** on the feature extraction + matching stages. The mapper (SfM + bundle adjustment) is CPU-only either way, so total speedup is more like **2–3×**.

Rough timings on our 93-image phone test (RTX 5070 Ti, 8 CPU threads):

| Setup | Total time | Feature extraction | Matching | Mapper |
|---|---|---|---|---|
| CPU SIFT | ~8 min | ~2.5 min | ~1.5 min | ~4 min |
| **GPU SIFT** | **~79 sec** | ~15 sec | ~10 sec | ~50 sec |

**Caveat:** GPU SIFT produces slightly different keypoints than CPU SIFT (different scale-space implementation). The mapper's seed-pair selection is also non-deterministic on GPU (floating-point ordering jitter in parallel reductions), so two GPU runs on the same data can produce different sub-model counts. With our current defaults (`matcher=exhaustive` + `single_camera=true`), all 93 images consistently end up in **one** sub-model — see the full test log further below.

---

## RAM usage and `num_threads`

CPU SIFT extracts features in parallel — each thread loads a different image at full resolution and builds its own scale-space pyramid. So **RAM scales linearly with thread count**. With 11 MP phone images and ~2 GB working set per thread (image + pyramid + feature buffers), the rough budget is:

| `num_threads` | Approx RAM (feature extraction step) | Speed |
|---|---|---|
| `-1` (all 12 cores) | ~25-29 GB ❌ fills 35 GB WSL2 | Fastest |
| `8` (default) | ~16-20 GB ⚠️ | ~1.5× slower than max |
| `6` | ~12-15 GB ✅ | ~2× slower |
| `4` | ~8-10 GB ✅✅ | ~3× slower |
| `2` | ~4-5 GB ✅✅✅ | ~6× slower |

**When using GPU SIFT**, the scale-space pyramid lives in VRAM (1–2 GB) instead of RAM, so RAM usage drops significantly — the mapper becomes the dominant RAM consumer. You can usually leave `num_threads=8`.

---

## Camera models

COLMAP supports several camera models with increasing complexity. The model defines **how 3D world points project onto the 2D image** — and which lens imperfections COLMAP tries to correct for. Below is a quick overview followed by per-model technical details.

| Model | # Params | Parameters | Best for |
|-------|----------|------------|----------|
| `SIMPLE_PINHOLE` | 3 | f, cx, cy | Repetitive textures (wheat, foliage), pre-undistorted images |
| `PINHOLE` | 4 | fx, fy, cx, cy | Non-square pixels, post-cropped images, distortion-corrected input |
| `SIMPLE_RADIAL` | 4 | f, cx, cy, k1 | Most consumer cameras with mild lens distortion (safe general default) |
| `OPENCV` | 8 | fx, fy, cx, cy, k1, k2, p1, p2 | Buildings, structured outdoor, indoor scenes — anything with rich geometry |
| `OPENCV_FISHEYE` | 8 | fx, fy, cx, cy, k1, k2, k3, k4 | Wide-angle / fisheye lenses only (GoPro, action cams, 180°+ FOV) |
| `FULL_OPENCV` | 12 | fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 | Specialised use cases, careful post-hoc calibration |

**The core trade-off:** each parameter is an unknown the bundle adjustment solves. More parameters → can fit lens distortion more accurately, **but** require more diverse feature matches to constrain. If the scene is repetitive (a wheat field is essentially identical at every spot), there isn't enough variation in the feature observations to uniquely solve those extra parameters, and the optimizer collapses or fails to register most images. **Less is often more.**

### Background: what these parameters actually mean

All COLMAP models start from the **pinhole projection equations**:
```
x_pixel = fx * X/Z + cx
y_pixel = fy * Y/Z + cy
```
where `(X, Y, Z)` is the 3D point in camera coordinates and `(x_pixel, y_pixel)` is its 2D image location.

- **`f` / `fx`, `fy` (focal length)** — controls field of view. Big `f` = narrow FOV (telephoto), small `f` = wide FOV (wide-angle). Measured in pixels (NOT mm — depends on sensor size). Separate `fx`/`fy` allow non-square pixels (rare in modern cameras but technically possible).
- **`cx`, `cy` (principal point)** — where the optical axis hits the sensor, in pixel coordinates. Usually within a few pixels of the image center. Offset matters when the lens is mechanically misaligned with the sensor or after cropping/cropped sensors.

The distortion parameters model how real lenses deviate from this perfect pinhole projection:

- **Radial distortion (`k1`, `k2`, ...)** — straight lines in the world become curved in the image. Cause: lens curvature varies with distance from the optical axis.
  - Positive `k1` → **barrel distortion** (lines bulge outward, common in wide-angle lenses)
  - Negative `k1` → **pincushion distortion** (lines pinch inward, common in zoom/telephoto)
  - `k2`, `k3`, ... add higher-order terms for stronger / more complex distortion
  - Math: `r_corrected = r * (1 + k1*r² + k2*r⁴ + k3*r⁶ + ...)`, where `r` is distance from the principal point
- **Tangential distortion (`p1`, `p2`)** — asymmetric distortion caused by the lens elements not being perfectly parallel to the sensor (manufacturing imperfections). Usually very small in modern cameras.

### Per-model technical details

**`SIMPLE_PINHOLE` — 3 params: `f, cx, cy`** (we use this for wheat)
The idealized pinhole camera: assumes a perfect lens with zero distortion and square pixels (`fx == fy`).
- **Why it's our default for wheat fields:** with only 3 unknowns to solve, the optimizer needs minimal constraint to converge. Repetitive vegetation provides poor feature variation, so we keep the model as constrained as possible. Modern phone cameras already correct most distortion in firmware (the JPEG you save is already mostly rectified), so the "no distortion" assumption is reasonable.
- **Limitations:** any residual lens distortion (barrel/pincushion) gets baked into the 3D points as small errors, slightly increasing reprojection error. For wheat fields where we get ~1.28 px reprojection error at 11 MP, this is negligible.

**`PINHOLE` — 4 params: `fx, fy, cx, cy`**
Same as SIMPLE_PINHOLE but allows independent horizontal and vertical focal lengths.
- **When this helps:** images that have been non-uniformly stretched or cropped (e.g. a 16:9 image that's been letterboxed and re-cropped), or sensors with rectangular pixels. Also useful when you want COLMAP to *measure* the pixel aspect ratio rather than assume 1:1.
- **Why it failed on our test (2/93):** adding `fy` as a free parameter doubles the focal-length ambiguity. On repetitive vegetation the cross-image consistency isn't strong enough to disambiguate `fx ≠ fy` from "different distance to scene," so the optimizer drifts and most images can't be registered.
- **Use when:** modern non-phone cameras with mild rectangular pixels, or post-rectified images.

**`SIMPLE_RADIAL` — 4 params: `f, cx, cy, k1`**
SIMPLE_PINHOLE plus a single radial distortion coefficient.
- **What it captures:** the dominant lens artifact in most consumer cameras — a single barrel/pincushion term. Mathematically simple, identifiable from even modest feature matches.
- **Why it's a "safe general default":** balances expressiveness (corrects most lens distortion) with identifiability (only 1 extra parameter to constrain). Works well for ordinary phone, DSLR, and webcam shots of varied scenes.
- **Use when:** unsure which model to pick and the scene has at least some geometric variety. Our wheat data is too repetitive even for this.

**`OPENCV` — 8 params: `fx, fy, cx, cy, k1, k2, p1, p2`**
The standard "computer vision" camera model — the same one OpenCV's `calibrateCamera()` produces.
- **What it captures:** two-term radial distortion (`k1` low-order, `k2` higher-order corrections) plus two-term tangential distortion (`p1`, `p2`) for misaligned lens elements. Plus independent `fx, fy`.
- **Why it's great for buildings / structured scenes:** rich geometry → many corner features at varied distances from the optical center → all 8 parameters are well-constrained → low reprojection error even with strong lens distortion.
- **Why it failed on our test (2/93):** 8 unknowns ÷ weak feature constraints = optimizer can't disambiguate. The radial and tangential terms become entangled with each other and with `fx, fy`. This is the classic over-parameterization failure mode.
- **Use when:** buildings, structured outdoor environments, indoor scenes, calibration targets — any scene with strong geometric features.

**`OPENCV_FISHEYE` — 8 params: `fx, fy, cx, cy, k1, k2, k3, k4`**
A **completely different projection model** for fisheye lenses (not just "more distortion params").
- **The math is different:** instead of the perspective projection `x = f*X/Z`, fisheye lenses use the **equidistant projection** `theta = atan(r_world/Z)`, then `r_image = f * theta`. This handles 180°+ FOV where the perspective math breaks down (would project infinitely).
- **Distortion terms** (`k1`, `k2`, `k3`, `k4`) are radial corrections in `theta` space, polynomial up to 9th order: `theta_distorted = theta + k1*theta³ + k2*theta⁵ + k3*theta⁷ + k4*theta⁹`.
- **Use when:** GoPro, action cams, dashcams with fisheye lenses, 360° rigs, anything with FOV > ~120°.
- **Don't use for:** normal phone cameras (their FOV ~70° is firmly in perspective territory) — would actually *hurt* reconstruction quality.

**`FULL_OPENCV` — 12 params: `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6`**
OpenCV's rational radial distortion model — six radial terms (`k1` through `k6`) plus two tangential.
- **What's new:** the radial distortion is modeled as a *rational* function `(1 + k1*r² + k2*r⁴ + k3*r⁶) / (1 + k4*r² + k5*r⁴ + k6*r⁶)`. Numerically more stable than just adding more polynomial terms for very strong distortion.
- **Use when:** very strongly distorted but **not fisheye** lenses (some specialised optical systems, scientific imaging, security camera lenses with heavy barrel distortion). Almost never the right choice for general photogrammetry.
- **Almost always overkill:** with 12 parameters, you need a *lot* of well-distributed feature matches to constrain everything. If you find yourself reaching for this, it's usually a sign you should pre-calibrate the camera once with a checkerboard target rather than asking SfM to figure it out from scene features.

### Summary heuristic

| If your scene is... | Use |
|---|---|
| Repetitive (wheat, foliage, grass, water, sand) | `SIMPLE_PINHOLE` |
| Mostly clean phone shots of regular scenes | `SIMPLE_RADIAL` |
| Buildings, indoor, structured outdoor | `OPENCV` |
| Wide-angle / fisheye lens (>120° FOV) | `OPENCV_FISHEYE` |
| Mixed phones (different makes/models) | `single_camera=false` + `SIMPLE_RADIAL` per camera |
| Strong distortion but normal lens | `FULL_OPENCV` (rare) |

For empirical comparison of camera models on real phone data, see the "Full empirical test log" section below.

---

## Full empirical test log

Series of tests run on plot `colmap_test_clean` (93 phone images, 11 MP, taken while walking through a wheat row). All times measured on RTX 5070 Ti + AMD Ryzen 7 7700X3D, 35 GB WSL2 RAM. Default camera `SIMPLE_PINHOLE` unless noted.

| # | SIFT | Uniform-sized | Matcher | `single_camera` | Sub-models | Images registered (largest) | Cameras | Total time | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CPU | ❌ (mixed dims) | exhaustive | ❌ | 1 | 63/93 | many | ~8 min | First baseline run, before any tuning |
| 2 | GPU | ❌ | sequential (25) | ❌ | 2 (29+66) | 29 (sub_0) | many | 79 sec | Mapper split by intrinsic group; undistorter took only sub_0 |
| 3 | GPU | ✅ | sequential (25) | ❌ | 2 (29+66) | 66 (sub_0) | 28 | 81 sec | Uniform helped pick the larger group first, but split persisted |
| 4 | GPU | ✅ | sequential (25) | ✅ | 2 (29+66) | 29 (sub_0) | 1 | 83 sec | One camera per sub-model — but the split itself didn't go away |
| **5** | **GPU** | **✅** | **exhaustive** | **✅** | **1** | **93/93** ✅ | **1** | **87 sec** | **Default config — all images bridged into one connected reconstruction** |
| 6 | GPU | ✅ | exhaustive | ❌ | 1 | 92/93 | 28 | 97 sec | Almost as good as #5, but 1 image dropped (intrinsics couldn't be solved per-image) and slower due to 28 cameras |

**Key takeaways:**
- **`matcher=exhaustive` is the critical fix** — it's what bridged the two halves of the walk into one reconstruction (tests #5, #6 vs #2, #3, #4)
- **`single_camera=true` is a useful refinement** — gets you 1 extra image registered, simplifies to 1 camera, ~10s faster (test #5 vs #6)
- **Uniform-sized images alone don't fix the split** (test #3 vs #2) — they help only when combined with the right matcher / single_camera
- **GPU SIFT gives ~6× speedup overall** (87 sec vs 480 sec for CPU baseline) and makes exhaustive matching essentially free vs sequential

**Camera-model results** (all on non-uniform images with CPU SIFT + exhaustive — earliest tests):
- `SIMPLE_PINHOLE`: **63/93** ✅ — only model that gave a usable reconstruction
- `PINHOLE`: 2/93 ❌
- `OPENCV`: 2/93 ❌

The extra distortion parameters in `PINHOLE` / `OPENCV` can't be constrained on repetitive wheat vegetation — the optimizer fails to find a unique solution and collapses.

---

## Exhaustive vs sequential matcher

- **`exhaustive`** (default) — matches every image against every other. O(N²) pairs. With **GPU SIFT** this is fast and cheap: only ~10 sec slower than sequential for 100 images, RAM stays low (matching runs in GPU batches). Robust against weak loop closures and walks that turn back on themselves.
- **`sequential`** — matches each image only against the next `sequential_overlap` images. Fast when matching is on CPU and N is large (~500+). Risky for walks with loop closure or mixed image groups — the mapper can split into disconnected sub-models if the overlap window misses cross-group matches.

**Why exhaustive is now the default:** empirically, on our 93-image phone test, sequential (overlap=25) split the reconstruction into two disconnected sub-models (29 + 66 images) because the two halves of the walk didn't have enough cross-matches. Switching to exhaustive bridged them into one model (93/93) with only ~6 sec extra runtime. See the test table below.

**When to switch back to sequential:** very large datasets (500+ images) where O(N²) image pairs become prohibitive even on GPU. For those, use `matcher=sequential sequential_overlap=40` (wider window than the default 25).

**Historical note about RAM:** old advice said "exhaustive matching can fill 30+ GB of RAM" — that was for **CPU** matching, where every batch loaded full descriptor sets into RAM. With GPU SIFT matching, descriptors live in VRAM (small — a few hundred MB) and RAM usage stays around 5–10 GB regardless of matcher choice.

---

## Logging and timing

The script prints each step with timing and saves the full COLMAP output to `{source_path}/logs/colmap.log`. The config used is snapshotted alongside in `convert_config.yaml`:

```
Step 1/3: Feature extraction...
  Feature extraction done in 52.1s
Step 2/3: Feature matching (exhaustive)...
  Feature matching done in 6.9s
Step 3/3: Mapping (SfM + bundle adjustment)...
  Mapper done in 23.3s
Undistorting images...
  Undistortion done in 0.3s
Done. Total time: 1.4 min (87s)
```

---

## Common issues

- **`"Single camera specified, but images have different dimensions"`** — your phone produced mixed-resolution images (HDR vs non-HDR). **Run `preprocess_uniform_size.py` first**, then re-run `convert.py` on the uniform output.
- **Only some images end up in `images/`** — COLMAP couldn't link all images into one connected reconstruction and only the largest sub-model was undistorted. Check `distorted/sparse/` — if there are folders `0/`, `1/`, `2/` etc., your images don't have enough overlap. Most common cause is mixed image dimensions (run `preprocess_uniform_size.py`). Other fixes: higher `sequential_overlap`, take photos with more overlap, or switch to `exhaustive` matching.
- **`"Finding good initial image pair"` appears multiple times in the log** — same as above, indicates disconnected sub-reconstructions.
- **CUDA-related errors** — the COLMAP we built supports CUDA 13.1 + RTX 5070 Ti (Blackwell, sm_120). If running on a different machine without CUDA, set `no_gpu=true` to fall back to CPU SIFT.
