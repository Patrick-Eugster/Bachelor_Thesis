# SfM Pipeline Comparison — Our COLMAP vs Supervisor's Agisoft

## Framing

The supervisor uses **Agisoft Metashape** for SfM on the phone data. Agisoft is a commercial closed-source tool with a paid license (~$3.5k node-locked). **We do not have access to Agisoft** — only the supervisor does, and we will not be installing it.

The thesis goal is therefore: **reproduce or beat Agisoft's reconstruction quality on phone images using only free, open-source tooling.** The supervisor's Agisoft `sparse/` folders (for fields A & D in the Demoanlage 2025 dataset) and the marker-error CSV are useful for one specific purpose: as a **reference ground truth to benchmark our pipeline against**. We use them to *measure* how close our free output is to Agisoft, not as a fallback to switch to. Everywhere else — new fields, future captures, anything in production — we must use the open-source pipeline.

The two pipelines being compared:

1. **Our pipeline (open-source, what we actually use):** `src/preprocessing/preprocess_uniform_size.py` + `src/preprocessing/run_colmap.py` (formerly `convert.py`), wrapped by `src/preprocessing/run_preprocessing.py` as a one-command orchestrator. Built around COLMAP, which we compiled from source with CUDA — see [`INSTALL_COLMAP_CUDA.md`](INSTALL_COLMAP_CUDA.md).
2. **Supervisor's pipeline (Agisoft, reference only):** `6-agisoft_preprocessing_demoanlage_2025.py` + `7-agisoft_compute_marker_errors.py` + `10_agisoft_calibration_quality_analysis.ipynb`. These three scripts were shared so we understand how the reference data was produced — they cannot be executed without a paid license.

Both ultimately produce a COLMAP-format `sparse/0/` + undistorted `images/` folder. The differences are in *how* they get there and *what extra information* their output carries.

---

## Does this affect 3DGS quality?

**Short answer: 3DGS training quality itself is fine with either pipeline.** Below are the three places where the difference between Agisoft and our COLMAP output actually shows up.

### 1. Phenotyping needs metric scale

3DGS is a **scale-invariant** representation. It learns gaussians at whatever scale your camera poses come in. Render quality (PSNR/SSIM/LPIPS), segmentation quality (FlashSplat), interactive viewer — none of those downstream steps read "real-world meters." So for those, COLMAP's arbitrary-unit `sparse/` is fine.

Where you hit a wall is **phenotyping**: when you eventually measure wheat head length, width, and volume, you need real units. Agisoft's metric `sparse/` lets you read `0.067 m = 6.7 cm` directly off the gaussian positions. With COLMAP's arbitrary units, a measured "0.04" could be 4 cm or 8 cm — meaningless until calibrated.

**How to close this gap with free tools:** see the "Closing the gap" section below.

### 2. Calibration accuracy affects render sharpness

Markers don't only give scale — they act as **constraints** during bundle adjustment. Each marker has a known real-world XYZ; the BA optimizer is pulled towards solutions where reconstructed marker positions match the surveyed ones. The supervisor's quality CSV shows that field A and D end up with mean 3D marker errors of ~16–18 mm and mean inter-marker distance errors of ~15 mm — very tight.

Our COLMAP run has no equivalent ground truth, so we can't directly measure how accurate the camera poses are. The looser camera poses show up in 3DGS as mild blur or ghosting rather than catastrophic failure — gaussians spread out to absorb the inconsistencies between views.

### 3. Markers help robustness on hard scenes

Wheat plots are repetitive and partially textureless. SIFT (the feature detector COLMAP uses) struggles in those regions because nearby keypoints look similar to each other → wrong matches → split reconstructions. This is exactly the sub-model-split problem we fought with on `colmap_test` earlier.

Coded markers solve this categorically: every marker has a unique 12-bit ID encoded in its dot pattern, so Agisoft never confuses one marker for another. They act as strong anchor features that always match correctly across views. **Agisoft basically never fails to register images on these scenes; COLMAP can.**

With our current defaults (uniform sizing + exhaustive matcher + single camera + SIMPLE_PINHOLE) we got 93/93 on `colmap_test_clean`. The next capture might be less lucky.

---

## Stage-by-stage comparison

Both pipelines do the same conceptual steps (load images → find features → match features → solve for cameras → triangulate 3D points → export). They differ in *how* each step is implemented and *what extra information* they use.

### SfM engine

**Our pipeline:** COLMAP 4.1.0, open-source, CLI-driven. Built from source with CUDA support. Free.

**Agisoft:** Commercial. Used via its Python API (`import Metashape`). Node-locked license. The Python module `Metashape` is **only available with a paid license**, which is why scripts 6 and 7 cannot be executed in our environment.

### Handling mixed image sizes

**Our pipeline:** Needs a separate preprocessing step (`preprocess_uniform_size.py`) that center-crops every image to the majority resolution. COLMAP groups images by intrinsic camera (image dimensions + EXIF); a mixed-resolution batch ends up split into disconnected sub-reconstructions. Phone HDR mode produces this mixed-resolution case constantly.

**Agisoft:** No preprocessing needed. Tolerates mixed image sizes natively — estimates per-image intrinsics where needed.

### Feature extraction

**Our pipeline:** COLMAP SIFT (Scale-Invariant Feature Transform). Detects ~5,000–20,000 keypoints per phone image at multiple scales. Runs on GPU via custom CUDA kernels — ~5–8× faster than CPU. Output: keypoint locations + 128-D descriptors per image, saved to a SQLite database.

**Agisoft:** Proprietary feature detector (similar in spirit to SIFT but not identical — Agisoft doesn't disclose the details). Runs at `downscale=0` ("use full resolution for feature detection"), the highest-quality setting. Slower than COLMAP's GPU SIFT but typically finds more usable features in low-texture regions.

### Image matching

**Our pipeline:** `exhaustive_matcher` — for every pair of images, compare their descriptors and find putative matches. O(N²) pairs but cheap per pair on GPU. For 93 images that's 4,278 pairs and takes ~10 s.

**Agisoft:** `ReferencePreselectionSequential` — uses each photo's EXIF GPS to pre-filter which pairs are even worth matching (only nearby photos in 3D space). This skips most of the O(N²) work. Phones write GPS into every photo's EXIF.

### Marker detection — the fundamental difference

**Our pipeline:** None. Whatever markers are visible in the images get treated as ordinary visual texture by SIFT. No special semantics.

**Agisoft:** `chunk.detectMarkers(tolerance=100)` — automatically detects the 12-bit coded circular targets in [`../Coded_12bit_15cm-square_13cm-outer-circle_.pdf`](../Coded_12bit_15cm-square_13cm-outer-circle_.pdf). Each marker has a unique binary ID printed as a ring of dots around it, so Agisoft can identify **which physical marker it sees in each image** with zero ambiguity. This gives perfect correspondences for those points across all views.

### Ground-truth coordinate constraints

**Our pipeline:** None. Bundle adjustment is unconstrained — solves for the most internally-consistent camera poses without any reference to the real world.

**Agisoft:** Imports each marker's surveyed XYZ from a CSV (Easting, Northing, Elevation in Swiss CH1903+/LV95, EPSG:2056). During bundle adjustment, the optimizer is pulled towards solutions where reconstructed marker positions match the surveyed coordinates. This fixes the reconstruction's **scale, orientation, and absolute position** in the real world.

### GPS and georeferencing

**Our pipeline:** Ignored.

**Agisoft:** Reads camera EXIF GPS (lat/lon/altitude) for every photo, reprojects to Swiss CH1903+/LV95 — the local Cartesian grid used for surveying in Switzerland. EPSG:2056 is the official Swiss coordinate system.

### Reconstruction scale — metric vs arbitrary

**Our pipeline:** **Arbitrary, unit-less.** COLMAP normalizes the reconstruction internally. A wheat head measured as 0.04 wide could be 4 cm or 8 cm in reality.

**Agisoft:** **Metric — real meters.** Because surveyed marker XYZ is used as a constraint, the entire reconstruction lands at correct scale. 0.04 m really is 4 cm.

### Camera model in the final export

**Our pipeline:** Step 1 of `run_colmap.py` extracts features using `SIMPLE_PINHOLE`. Step 4 (`image_undistorter`) then rewrites images and intrinsics to ideal pinhole. Final `sparse/0/cameras.bin` contains a `PINHOLE` camera with no distortion params; `images/` contains undistorted JPEGs.

**Agisoft:** Internally estimates a much richer distortion model during alignment (typically full Brown-Conrady: radial k1/k2/k3 + tangential p1/p2). On export, `convert_to_pinhole=True` undistorts images and re-writes intrinsics as plain pinhole. Final output is also pinhole+undistorted.

The difference: Agisoft's undistortion is more accurate because it started from a better distortion estimate. Ours is "good enough" — SIMPLE_PINHOLE is empirically what works on wheat (PINHOLE and OPENCV both failed in our tests).

### Quality evaluation

**Our pipeline:** None built-in. We can eyeball whether the right number of images registered, but we have no quantitative measure of camera-pose accuracy.

**Agisoft:** Script 7 (`7-agisoft_compute_marker_errors.py`) opens each `.psx` and computes per-session:
- **Reprojection error (px)** — average pixel distance between observed marker positions in images and where reconstructed 3D markers project back to.
- **3D error (m)** — distance between reconstructed and surveyed marker positions, in meters.
- **Inter-marker distance error (m)** — for every marker pair, compare reconstructed inter-marker distance to ruler-measured distance from `Demoanlage-2025-markers-manual-distances.xlsx`. Independent of GPS quality.

Written to `marker_errors_summary.csv`. Notebook 10 visualizes the CSV. The threshold-based flagging found **72 of 148 sessions** exceed at least one quality threshold — so even Agisoft has bad days.

### Determinism

**Our pipeline:** GPU SIFT is non-deterministic at the bit level. ~99.9% identical between runs on most scenes; can differ on borderline scenes. Set `no_gpu=true` for bit-exact reproducibility at ~5–8× slower.

**Agisoft:** Essentially deterministic per settings.

### Configuration

**Our pipeline:** Hydra YAML configs + short CLI overrides like `plot=20250618`. All defaults documented in [`../src/preprocessing/README.md`](../src/preprocessing/README.md).

**Agisoft:** Hardcoded paths inside `main()` of each script. Designed for one user, one machine, one dataset layout.

### Runtime

**Our pipeline:** ~80–90 s on RTX 5070 Ti for 93 phone images at ~12 MP each.

**Agisoft:** Minutes per session in highest-quality mode.

---

## Closing the gap with open-source tools

This is the actionable part. The thesis goal is to match Agisoft's quality with free tooling — here are concrete options for each gap.

### Gap 1: Metric scale

**Option A — Post-hoc scale calibration (easiest, no code changes).** Measure one known real-world distance in the scene (e.g. the printed marker is exactly 15 cm × 15 cm, or measure two marker centers with a ruler). After COLMAP finishes, compute the same distance in the reconstruction, divide → get a scale factor, and apply it to all gaussian positions before phenotyping. This is what we'd do as a baseline. The marker PDF says **15 cm square outer side, 13 cm inner circle** — that's a built-in scale bar in every image.

**Option B — COLMAP GCP support (more accurate).** COLMAP recently added Ground Control Point support via bundle adjustment — see [colmap/colmap#593](https://github.com/colmap/colmap/issues/593#issuecomment-3926658343). You provide a list of `(image_name, x_pixel, y_pixel, X_world, Y_world, Z_world)` rows; COLMAP uses them as constraints in BA, exactly like Agisoft does. This requires detecting marker positions in each image first (next gap).

**Option C — full pycolmap pipeline with markers integrated from the start.** `pycolmap` is the official Python bindings to the same COLMAP C++ codebase (same algorithms, same CUDA when built with CUDA flags). Instead of bolting GCPs onto a finished reconstruction post-hoc, rewrite the preprocessing pipeline as Python so markers are first-class citizens throughout, mirroring what Agisoft does internally:

| Step | What |
|---|---|
| 1 | SIFT feature extraction (CUDA) |
| 2 | **ArUco detection** via `cv2.aruco.detectMarkers` — get marker pixel coords per image |
| 3 | Feature matching (CUDA) |
| 4 | **Inject marker correspondences into the matches database** — same marker ID in two photos = a correspondence, just like a SIFT match |
| 5 | Mapper — uses SIFT + marker matches together |
| 6 | Bundle adjustment with marker world coordinates as **fixed GCPs** |
| 7 | Undistort |

This is the *Agisoft-equivalent* workflow. Adding markers post-hoc (a "mix" of CLI + pycolmap, CLI does 1–5, pycolmap re-runs BA with GCPs in step 6) sounds simpler but doesn't work as well: by the time the CLI mapper finishes, camera poses are already fixed; post-hoc BA can refine but can't undo a bad initial loop closure or wrong scale guess. Markers need to guide the search from the start.

**Build/install note:** `pip install pycolmap` from PyPI ships CPU-only wheels — for CUDA-accelerated SIFT, build pycolmap from source with the same flags as our existing CUDA COLMAP CLI (see [`INSTALL_COLMAP_CUDA.md`](INSTALL_COLMAP_CUDA.md) — most of the dependency work is already done since we built CUDA COLMAP from source on this machine).

**Rollback plan if pycolmap doesn't pan out:** our CUDA-enabled COLMAP CLI is independent. If full-pycolmap turns out to be a dead end (build issues, missing API surface, etc.), we don't lose anything — we can keep using the CLI we already have via [`run_colmap.py`](../src/preprocessing/run_colmap.py), and fall back to Option A (post-hoc scale calibration) for metric scale. The CUDA CLI is the safety net.

### Gap 2: Marker detection (and using markers as constraints)

The 12-bit Agisoft markers are a proprietary format, but **OpenCV has built-in support for ArUco markers**, which are the open-source equivalent. ArUco is:
- A library of pre-generated coded markers (`cv2.aruco`).
- Used heavily in robotics and AR.
- Detects markers and their unique IDs in milliseconds per image.
- Free, MIT-licensed.

If we re-print our own marker set as ArUco and re-shoot future captures with them, we can write a ~50-line script that runs `cv2.aruco.detectMarkers()` on every image, builds a list of GCP correspondences, and feeds them to COLMAP via option B above. **For the existing Demoanlage data we can't do this** (the supervisor's markers are Agisoft-format), but it's the obvious upgrade path for any new captures we make.

If we want to use the *existing* Agisoft markers as constraints: the supervisor mentioned they could give us the 3D positions of the marker centers if needed. We could then manually annotate the marker pixel positions in a few images, or write a custom detector for the 12-bit pattern. More work, smaller payoff.

### Gap 3: Robustness on textureless / repetitive scenes (wheat)

This is where COLMAP+SIFT genuinely lags Agisoft, and it's also where the most exciting open-source progress has happened in the last 3 years. **Deep-learning feature detectors** trained on millions of image pairs are now strictly better than SIFT for hard scenes:

- **SuperPoint + SuperGlue** (Magic Leap, 2020) — current gold standard. Drops into the COLMAP pipeline via [hloc (HierarchicalLocalization)](https://github.com/cvg/Hierarchical-Localization). Often turns "53/93 registered" into "93/93 registered" on hard scenes. We have GPU, this should work.
- **LightGlue** (ETH, 2023) — faster successor to SuperGlue, similar quality. Same hloc integration.
- **DISK**, **ALIKED**, **DeDoDe** — alternative detectors, sometimes better than SuperPoint on specific scene types.
- **VGGSfM / MASt3R-SfM** (2024) — end-to-end neural SfM, no SIFT at all. Newer and less proven but potentially huge gains on hard scenes. Still research-grade.

**Practical step:** if our current SIFT-based COLMAP starts failing on a phone capture, the first thing to try is SuperPoint + SuperGlue via `hloc`. That alone often closes the gap to Agisoft on textureless scenes. It's been on the "should try" list for a while.

### How hloc + ArUco GCPs combine

The two upgrades are **orthogonal** — they fix different problems and compose cleanly:

| Lever | Problem it fixes | Where in the pipeline |
|---|---|---|
| SuperPoint + SuperGlue (via hloc) | SIFT fails on repetitive / textureless scenes → low registration rates | Replaces steps 1 + 3 (features + matching) |
| ArUco + GCPs (via pycolmap) | No metric scale, no georeferencing | Adds steps 2 + 4 + 6 (marker detect + inject + GCP-aware BA) |

hloc writes its SuperPoint features and SuperGlue matches into the **same COLMAP database format** that pycolmap reads natively, so they slot together without glue code. Combined pipeline:

| Step | Tool | What |
|---|---|---|
| 1 | **hloc (SuperPoint)** | Deep feature extraction — beats SIFT on wheat |
| 2 | OpenCV (`cv2.aruco`) | ArUco marker detection per image |
| 3 | **hloc (SuperGlue / LightGlue)** | Deep feature matching — beats default matchers |
| 4 | pycolmap | Inject marker correspondences into matches DB |
| 5 | pycolmap | Mapper — uses SuperPoint features + marker matches |
| 6 | pycolmap | Bundle adjustment with marker world coords as fixed GCPs |
| 7 | pycolmap | Undistort |

You can apply either upgrade independently:
- If 3DGS metrics look fine but you need metric scale → just add ArUco/GCPs (steps 2, 4, 6).
- If registration rates are low (sessions producing fragmented sub-models) → just add hloc (steps 1, 3).
- If both are problems → the full combined pipeline above.

CUDA-enabled COLMAP CLI remains the safety net for all variants — if either hloc or pycolmap turn out to be problematic, fall back to the current SIFT-based CLI pipeline.

### Gap 4: Quality evaluation

Even without our own markers, we can build the same evaluation Agisoft has — *as long as we have the supervisor's `sparse/` as ground truth for at least some sessions.* The script would:

1. Take our COLMAP `sparse/` and the supervisor's Agisoft `sparse/` for the same capture.
2. Align them with a 7-DOF similarity transform (Umeyama).
3. Compute per-camera pose error (translation in meters, rotation in degrees).

This gives us a **per-capture benchmark** of "how close did COLMAP get to Agisoft." We can use it to validate any future improvements (e.g. switching to SuperPoint+SuperGlue should reduce these errors).

**Status: implemented.** See [`../src/preprocessing/compare_to_agisoft.py`](../src/preprocessing/compare_to_agisoft.py) — runs Umeyama alignment, reports mean/median/max translation (mm) + rotation (deg), and writes a per-camera JSON. For interpreting the supervisor's own quality CSV (`marker_errors_summary.csv`) when picking which sessions to benchmark against, see [`AGISOFT_QUALITY_METRICS.md`](AGISOFT_QUALITY_METRICS.md).

---

## What this means in practice for the thesis

1. **Use supervisor's `sparse/` for fields A & D ONLY as a quality benchmark.** Compare our COLMAP output against theirs to understand how far off we are. Do not depend on Agisoft `sparse/` for anything downstream — the thesis story has to work end-to-end without it.

2. **Our pipeline is the primary one.** Iterate on it until our reconstruction quality is "close enough" to Agisoft. The empirical bar: 100% registration rate on `colmap_test_clean` is already there; the next things to verify are pose accuracy (via Gap 4 evaluation) and 3DGS render quality on COLMAP-only data.

3. **For phenotyping (later in the thesis):** plan for post-hoc scale calibration using the 15-cm printed marker size. This is the simplest free path to metric units.

4. **If captures start failing:** the upgrade path is SuperPoint + SuperGlue via hloc, not switching to Agisoft.

---

## Where the two pipelines are equivalent (so 3DGS just works)

Once both have produced their output, 3DGS doesn't care which one made the `sparse/`:

- **File format** — both produce COLMAP `sparse/0/`.
- **Camera model** — both end up as `PINHOLE` with no distortion params.
- **Images** — both are pre-undistorted JPEGs matching the pinhole intrinsics.
- **3DGS code path** — identical, no branching.

The choice of pipeline only affects *what gets into the `sparse/`*, not what 3DGS does with it.

---

## Reading the supervisor's scripts (reference only — cannot be executed)

| Script | What it does | Why you'd read it |
|---|---|---|
| `6-agisoft_preprocessing_demoanlage_2025.py` | The Agisoft SfM pipeline: addPhotos → detectMarkers → CRS reproject → import marker GT → matchPhotos → alignCameras → exportCameras (COLMAP format) | Understand what settings produced the `sparse/` you received |
| `7-agisoft_compute_marker_errors.py` | Opens each `.psx`, computes per-marker 3D error, reprojection error, and inter-marker distance error vs ruler measurements; writes `marker_errors_summary.csv` | Understand exactly what each column in the CSV means |
| `10_agisoft_calibration_quality_analysis.ipynb` | Pure pandas/matplotlib on `marker_errors_summary.csv` — heatmaps, time-series, "worst sessions", composite quality score | **Runnable** if you have the CSV — use it to pick which sessions to benchmark against |

The notebook is the only one you can actually execute (no `Metashape` import needed); the other two require a paid Agisoft license.
