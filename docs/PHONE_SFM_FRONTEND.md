# Phone SfM front-end: closing the COLMAP-vs-Agisoft registration gap

**TL;DR** — On repetitive phone wheat imagery, our COLMAP (SIFT) fragmented `field_A/20250603` into **4 disconnected sub-models** (largest = 76 of 113 images) while Agisoft put all **113 in one model**. The gap was the **feature front-end**, not the mapper. Switching to **ALIKED + LightGlue** (learned detector + learned matcher, both native in COLMAP 4.1) yields **one connected model with all 113/113 images**, denser than Agisoft. **Sequential** matching gives the same result ~8× faster, so it's the default for our continuous sweeps. Both SIFT and ALIKED are selectable in `run_colmap.py` (`front_end=sift|aliked`).

---

## 1. The symptom

`run_colmap.py` reported **76/113 registered** on `field_A/20250603` — Agisoft registered all 113 of the *same* images. Earlier hypotheses ("green wheat", too few markers) were wrong: Agisoft proves the images are fine.

## 2. The diagnosis — fragmentation, not failure

Our COLMAP actually registered **110/113** images, but the incremental mapper split them into **4 disconnected sub-models** (76 + 12 + 11 + 11); `run_colmap.py` keeps only the largest. The fragments are **temporally interleaved** (all inside one continuous 150300→150432 sweep — e.g. sub-model 3 covers 150341–150349, *inside* model 0's span), so they cover the same physical ground. The mapper repeatedly lost and re-acquired the thread on the repetitive canopy, spawning a new sub-model each time.

A diagnostic scorer — **`src/analysis/analyze_sfm_connectivity.py`** (read-only) — measures three things for any COLMAP run:

- **A. Coverage** — sub-models, largest model, total registered (from the sparse model).
- **B. Connectivity** — geometrically-verified image pairs, match-graph density, median inliers/pair, and the **connected components of the match graph** (the single best predictor of whether SfM can build one model), from `database.db`.
- **C. Quality** — #3D points, mean track length, mean reprojection error (from the sparse model).

```
python src/analysis/analyze_sfm_connectivity.py --model <sparse_dir> [--database <db>] --label NAME --n-input 113
```

### Why the mapper isn't to blame
- `colmap model_merger` **fails** — the sub-models share *zero* co-registered images, so there's nothing to align on.
- `colmap image_registrator` **adds 0** — the missing frames see only 9–21 of model 0's 3D points (≪ the ~30 needed).
- Re-running the mapper with `--Mapper.multiple_models 0` (forbid fragments) just **drops the 34 frames to give 76** — it can't conjure registrations that the matches don't support.
- The database shows the bottleneck: between an 11-frame fragment and the 76-frame model there are only **10 verified pairs** (of 836), and just **4** above the registration bar.

→ Root cause: **sparse features → sparse triangulation → marginal frames have no 2D-3D points to register against.** That's a feature-front-end problem.

## 3. The fix — ALIKED + LightGlue

COLMAP 4.1 ships, natively (no hloc/install), a learned detector (**ALIKED**) and a learned matcher (**LightGlue**):
`--FeatureExtraction.type ALIKED_N16ROT` + `--FeatureMatching.type ALIKED_LIGHTGLUE`.

### Benchmark on `field_A/20250603` (113 images, SIMPLE_PINHOLE, single_camera)

| metric | SIFT (baseline) | ALIKED+LG exhaustive | **ALIKED+LG sequential** | Agisoft (gold) |
|---|---|---|---|---|
| **sub-models** | **4 (fragmented)** | 1 | **1** | 1 |
| **largest model** | **76 / 113** | 113 / 113 | **113 / 113** | 113 / 113 |
| 3D points | 4,472 | 62,351 | **55,593** | 49,592 |
| observations / image | 202 | 1,474 | **1,304** | 900 |
| verified pairs | 1,043 | 1,794 | 427 | — |
| median inliers / pair | 1 | 22 | **763** | — |
| reprojection error | 1.31 px | 1.39 px | **1.32 px** | — |
| **matching time** | 18 s | 742 s (~12 min) | **94 s** | — |
| **total run** | ~105 s | ~896 s (~15 min) | **~229 s (~3.8 min)** | — |

**Reading it:** ALIKED's denser, better-localized learned features + LightGlue's ambiguity-resolving matcher fix the fragmentation exactly as the diagnosis predicted — one connected model, **denser than Agisoft**, healthy reprojection. **Sequential** matching keeps far fewer pairs (427 vs 1,794) but each one is much stronger (median 763 vs 22 inliers) — it spends effort only on the neighbour pairs that actually overlap, which is right for a continuous sweep. Same result, ~8× faster matching.

## 4. Using it — `run_colmap.py`

Selectable via config (`configs/preprocessing/colmap.yaml`); **SIFT stays the default and is byte-identical to before.**

```bash
# SIFT (default, unchanged)
python src/preprocessing/run_colmap.py field=field_A plot=20250603

# ALIKED + LightGlue, sequential matching (recommended for ordered sweeps)
python src/preprocessing/run_colmap.py field=field_A plot=20250603 \
  front_end=aliked aliked_cuda12_libdir=/path/to/cuda12libs

# ALIKED with exhaustive matching (unordered image sets)
python src/preprocessing/run_colmap.py field=field_A plot=20250603 \
  front_end=aliked matcher=exhaustive aliked_cuda12_libdir=/path/to/cuda12libs
```

Config knobs (ALIKED only): `aliked_max_num_features` (4096), `aliked_max_image_size` (**default 2048** — feature-detection resolution; output images stay full-res; drop to 1600 for more VRAM headroom — see §5b), `aliked_extract_threads` (4 — caps the CPU-side decode), `aliked_cuda12_libdir` (see below).

**Matcher default:** `exhaustive` (flipped back from sequential on 2026-06-28). Sequential and exhaustive give the *same registration* (1 model, 100 %), but sequential leaves the single linear walk without loop closure → **camera-pose drift at the sweep endpoints**. Exhaustive's all-pairs cross-links close the open walk and materially improve pose accuracy + marker scale (14-session study: median pose error vs Agisoft down ~30–80 %, 13/14 sessions reliable metric). Use `matcher=sequential` (with a higher `sequential_overlap`) only for **large sets (170+ imgs)** to dodge exhaustive's O(N²) cost, or for **unordered** image sets. Full write-up: [PHONE_SFM_POSE_ACCURACY.md](PHONE_SFM_POSE_ACCURACY.md). Applies to both front-ends.

## 5. Environment gotchas (important)

### 5a. CUDA 12 vs 13 — ALIKED GPU aborts without a lib shim
This COLMAP's ONNX-Runtime GPU provider was compiled against **CUDA 12**, so on a **CUDA-13** box ALIKED aborts with `libcublasLt.so.12: cannot open shared object file`. Fix — **without touching the system or any Python/torch env** — supply the CUDA-12 libs in a folder and point the config at it:

```bash
pip install --target /some/stable/cuda12libs \
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12
# then: run_colmap.py ... front_end=aliked aliked_cuda12_libdir=/some/stable/cuda12libs
```

`run_colmap.py` walks that dir for `*/lib` subdirs and prepends them to `LD_LIBRARY_PATH` for the COLMAP calls. Leave `aliked_cuda12_libdir=""` on a box where the system CUDA already matches (e.g. a proper CUDA-12 install, or build/point at a matching onnxruntime). **~1.5 GB download** — put it somewhere stable, not a temp dir.

### 5b. Resource profile (measured)
| phase | GPU VRAM | CPU | duration |
|---|---|---|---|
| **ALIKED extraction** | **scales with resolution** — see table below; ~15.7 GB at full 12 MP | **all cores** by default — CPU-side JPEG decode + pre/post-proc | ~45 s (GPU) |
| **LightGlue matching** | ~4 GB (operates on the descriptors, not images) | light | seq ~94 s / exh ~12 min |

**VRAM is the gotcha.** ALIKED is a dense CNN, so one image's intermediate tensor scales with **pixel count** — at full 12 MP it needs ~15.7 GB, which **OOMs a 16 GB card unless it's nearly empty** (it only fit on the very first run because the GPU was idle; with ~2 GB used elsewhere it fails). Fix = **`aliked_max_image_size`** (default **1600**), which downscales the image **for feature detection only**:

| `aliked_max_image_size` | peak VRAM (12 MP source) | result |
|---|---|---|
| 1024 | 8.2 GB | safe, fewer features |
| 1600 | 10 GB | safe headroom; registered 93/93 on field_A/20250618 |
| **2048 (default)** | **~15 GB** | completed 13/13 sessions, all 100 % in one model; spike-risky on a shared GPU |
| 0 / 3200 | ~15.7 GB | OOM unless GPU empty |

The jump is sharp (bimodal): ≤1664 ≈ 9–10 GB, ≥1728 ≈ 15 GB — the ONNX arena reserves in power-of-two chunks, so there's no stable 12–14 GB option. **2048 is the default** (more features, validated across all sessions); use **1600** if the GPU is shared and you want OOM-spike margin. For full-res *features* with zero VRAM risk, run extraction on CPU (`no_gpu`-style: ONNX CPU provider uses system RAM) + matching on GPU — slower (~6–8 min/session) but spike-proof.

**Crucially, this does NOT shrink your output images.** `image_undistorter` re-reads the originals and writes `images/` at **full resolution** (verified 4032×3024 with `aliked_max_image_size=1600`) — only the keypoint *detection* for pose estimation runs at the reduced size, exactly like Agisoft's "High" (vs "Highest") alignment. So you get full-res images **and** safe VRAM. `aliked_max_num_features` barely affects the peak (the dense feature map is computed before keypoint selection).

- **CPU:** COLMAP's own thread pool ignores `OMP_NUM_THREADS`; the only thing that caps the extraction decode is `--FeatureExtraction.num_threads` (wired as `aliked_extract_threads`, default 4). For a *hard* core cap, additionally launch under `taskset -c 0-3`.

## 6. Status & next steps
- ✅ Diagnosed (fragmentation from sparse cross-cluster matching), scorer built, ALIKED+LightGlue fix validated on `field_A/20250603`, wired into `run_colmap.py` (SIFT kept as baseline).
- ✅ **Generalises across the whole phone series.** Ran ALIKED (2048-px feature extraction, sequential, full-res output) on all 13 other sessions: **every one → 1 connected model, 100 % of images registered** (both fields, 64→183 images, incl. the Pixel-6a camera). The lone SIFT session left untouched (`field_A/20250603`, 4 fragments / 76-of-113) is the side-by-side baseline.
- ☐ Pose-accuracy check of the ALIKED model vs Agisoft (`compare_to_agisoft.py`) — registration is solved, this confirms the geometry is also right.
- ☐ Decide whether to flip the **front-end** default to ALIKED once it's validated broadly (currently `sift` to preserve the established baseline).
- ✅ Made the CUDA-12 lib dir permanent: copied to in-repo **`tools/cuda12libs/`** (~2.5 GB, gitignored) and set as the **default `aliked_cuda12_libdir`** in `configs/preprocessing/colmap.yaml` — replaces the old ephemeral scratchpad path that every prior run referenced. Set `aliked_cuda12_libdir=""` on Euler (system CUDA matches).

Diagnosis history and the marker context: [MARKER_INTEGRATION_PLAN.md](MARKER_INTEGRATION_PLAN.md) item (b). Scorer: `src/analysis/analyze_sfm_connectivity.py`.
