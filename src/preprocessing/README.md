# Phone SfM Preprocessing — `src/preprocessing/`

Turns raw phone captures of a wheat plot into the undistorted `images/` and the `sparse/0/`
reconstruction that the rest of the pipeline builds on. It runs Structure-from-Motion (SfM) with
COLMAP to recover the camera poses and calibration, undistort the images, and produce a sparse
point cloud. The undistorted images feed the following pipeline stages, and the recovered camera
poses and sparse point cloud are the specific input for 3D reconstruction with 3D Gaussian
Splatting (3DGS). This is our open-source SfM path, so the phone pipeline does not depend on any
commercial photogrammetry software.

This stage is **phone only**. FIP plots already come processed by Agisoft in COLMAP format, so
they skip preprocessing entirely.

Everything is driven by the orchestrator `run_preprocessing.py`, which runs the steps below in
order and shares `field=` and `plot=` across all of them:

```bash
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715
```

## The steps

1. **Uniform size** (`preprocess_uniform_size.py`) — many phones save slightly different image
   sizes within one capture session, and COLMAP then treats each size as a separate camera and
   can split the reconstruction into disconnected pieces. This step center-crops every image to
   the majority size, so it only trims a few border pixels and leaves the focal length and
   optical center untouched. If all images already share one size, it links `input_uniform/` to
   `input/` at no disk cost, so the step is always safe to run.

2. **COLMAP SfM** (`run_colmap.py`) — runs feature extraction, matching, the SfM mapper, and
   image undistortion. By default it uses the ALIKED + LightGlue front end, the OPENCV camera
   model, and exhaustive matching (see [Key options](#key-options)). The mapper occasionally
   spawns a small stray sub-model next to the real one, so the script always undistorts the
   largest connected model. The result is `images/` (undistorted) and `sparse/0/` (poses and
   points), ready for the following pipeline stages.

3. **Compare to Agisoft** (`compare_to_agisoft.py`, optional) — aligns our `sparse/0/` to an
   Agisoft reference with Umeyama and reports the per-camera translation and rotation error. It
   only runs when an `agisoft/` reference is present in the session folder, so most users can
   leave it off.

## How to run

```bash
# default: uniform size + COLMAP (compare off), cleans stale COLMAP output first
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715

# also compare to Agisoft (needs agisoft/sparse/0/ in the session folder)
python src/preprocessing/run_preprocessing.py field=field_A plot=20250618 run_compare=true

# skip a step, for example when the images are already uniform
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715 run_uniform=false
```

By default the orchestrator wipes stale COLMAP output before re-running, so a re-run always
starts clean. Pass `clean_before_colmap=false` to keep a previous run's output.

Each step can also be run on its own (`preprocess_uniform_size.py`, `run_colmap.py`,
`compare_to_agisoft.py`), which is handy when debugging one stage.

## Key options

Pass these to `run_colmap.py`, or set them in `configs/preprocessing/colmap.yaml`. The
orchestrator forwards only `field=` and `plot=`, so the SfM options below go on `run_colmap.py`
directly.

| Option | Default | Meaning |
|--------|---------|---------|
| `front_end` | `aliked` | Feature front end. `aliked` = learned ALIKED detector + LightGlue matcher. `sift` = classic COLMAP SIFT |
| `camera` | `OPENCV` | Camera model. `OPENCV` models the lens distortion, `SIMPLE_PINHOLE` is the lighter crop-free fallback |
| `matcher` | `exhaustive` | `exhaustive` matches all image pairs, `sequential` matches each frame to its time neighbours |
| `single_camera` | `true` | Solve one shared set of intrinsics for all images. Needs uniform image sizes |
| `image_subdir` | `input_uniform` | Which subfolder COLMAP reads. Set `input` to skip the uniform-size step |
| `num_threads` | `8` | Threads for extraction and matching. Lower uses less RAM |
| `no_gpu` | `false` | Force CPU features. Only needed if your COLMAP build lacks CUDA |
| `variant_dir` | `""` | Write COLMAP output into a subfolder instead of overwriting the baseline, for SfM A/B tests |

These are the options you are most likely to change, not the full list. The config file
holds the rest.

A few defaults are worth explaining, since they carry the phone pipeline:

- **ALIKED + LightGlue** keeps repetitive wheat in one connected model where **SIFT** fragments
  it into several. On one session SIFT produced four sub-models with the largest holding 76 of
  113 images, while ALIKED registered all 113 in a single model.
  ```bash
  python src/preprocessing/run_colmap.py field=field_A plot=20250715 front_end=sift
  ```

- **OPENCV** models the real lens distortion of the phone camera. With the strong ALIKED matches
  every camera model registers fully on wheat, so OPENCV is the marginally most accurate choice.
  `SIMPLE_PINHOLE` is a lighter fallback that does not crop the undistorted image.
  ```bash
  python src/preprocessing/run_colmap.py field=field_A plot=20250715 camera=SIMPLE_PINHOLE
  ```

- **Exhaustive matching** gives the walk loop closure, which cuts camera-pose drift at the ends
  of the sweep. For very large sessions the all-pairs cost grows quickly, so prefer
  `matcher=sequential` there.
  ```bash
  python src/preprocessing/run_colmap.py field=field_A plot=big_session matcher=sequential
  ```

## ALIKED requirements

The ALIKED front end needs a GPU and a set of CUDA-12 onnxruntime libraries. By default
`aliked_cuda12_libdir` points at `tools/cuda12libs/` (about 2.5 GB, gitignored, so not present in
a fresh clone). Two ways out if you do not have them:

- set `aliked_cuda12_libdir=""` on a machine whose system CUDA already matches, or
- set `front_end=sift` to use the classic SIFT path, which needs no extra libraries.

## Outputs

Written into the session folder:

```
input_plots/phone/<field>/<session>/
├── input_uniform/   ← uniform-size images from step 1 (or a symlink to input/)
├── images/          ← undistorted images used by the following pipeline stages
├── sparse/0/        ← camera poses and sparse points, the 3DGS input (cameras/images/points3D.bin)
├── distorted/       ← COLMAP working files (safe to delete after)
├── stereo/          ← created by the undistorter, not used by our pipeline
└── logs/            ← colmap.log + per-step summary JSON files
```

The orchestrator re-prints a summary block for each step at the end, followed by a timing table,
so the earlier summaries are not buried under COLMAP's output.

## Diagnostic

`analyze_sharpness.py` scores every image by Laplacian variance and reports the sharpness
distribution of a session. It only reads the images and never changes them, so use it to check
whether a poor COLMAP result comes from blurry input rather than the pipeline.

```bash
python src/preprocessing/analyze_sharpness.py field=field_A plot=20250715
```

A few more optional checks on an SfM model live in [`src/analysis/`](../analysis/) and also run on
their own:

- **`analyze_sfm_connectivity.py`** — tells you whether COLMAP tied all your images into one
  connected model or quietly dropped and split some.
- **`analyze_sparseness.py`** — measures how many cameras see each point and from how many
  directions, which says more about a capture than the plain image count.
- **`fip_principal_point_offset.py`** — measures how far each camera's principal point sits from
  the image center.
- **`run_hloc_sfm.py`** — builds a model with an alternative SfM front-end, for a comparison
  against the default one.

Read the notes at the top of [`src/analysis/README.md`](../analysis/README.md) before running one.

## Coded markers

The phone captures also carry coded ground markers, handled by a separate set of steps under
[`markers/`](markers/) that are off by default. They build the marker-based region of interest
and, for phenotyping, the metric scale. See [`markers/README.md`](markers/README.md).

> The deeper SfM write-ups (camera-model and front-end benchmarks, pose-accuracy study) live
> under `docs/`, which is a private working-notes submodule and is not part of a public clone.
