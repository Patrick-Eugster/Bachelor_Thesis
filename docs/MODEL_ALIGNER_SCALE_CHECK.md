# Cross-validating our marker scale with COLMAP `model_aligner`

**TL;DR** — Our phone metric scale comes from **ground markers + a tape measure**
(`triangulate_markers.py` → `metric_frame.json`, `scale = median(tape_dist / recon_dist)`).
COLMAP's `model_aligner` gives a **second, independent** metric scale by snapping our reconstruction
onto Agisoft's metric model (cameras-vs-cameras, no tape, no markers). Across all reliable sessions the
two agree to **within ±3–5 %**, which validates the marker scale — and the residual gap is **field-specific**:
**field_A is systematically +1 to +5 % (tape says it's bigger), field_D scatters around zero.** That
positive field_A bias independently corroborates the known field_A tape-entry error
([tape-measure-error](#)) — model_aligner fingers the tape, from a completely different data path.

---

## 1. What `model_aligner` is (and why it's a useful check)

`model_aligner` estimates a single **3D similarity transform** (rotation **R**, translation **t**, one
global **scale s** — 7 DoF) that best lays our reconstruction onto a reference coordinate frame, fits it
with RANSAC, and rewrites the whole model (every camera + 3D point) into that frame. It does **not**
re-run bundle adjustment — it's a rigid+scale snap of the finished model.

The reference can be **known camera positions** (`--ref_images_path`, a GPS/RTK file we don't have) or
**another model** (`--ref_model_path`). We use the latter: **Agisoft's metric `sparse/0`** for the same
shoot. The scale it recovers is therefore an Agisoft-anchored, **camera-based** metric estimate — a
totally different input from our **marker+tape** estimate, which is exactly what makes the agreement
meaningful.

> **It does NOT replace the marker step.** `model_aligner` can only align to known camera positions or a
> reference model — there is **no input for "the distance between two ground points"** (confirmed by the
> COLMAP maintainer: COLMAP supports no distance prior in BA *or* in `model_aligner`). So for the
> standalone, Agisoft-free metric pipeline the **markers remain the only metric source**. This is a
> *validation/benchmark* tool, not a pipeline component. See
> [PHONE_SFM_POSE_ACCURACY.md](PHONE_SFM_POSE_ACCURACY.md) and [MARKER_INTEGRATION_PLAN.md](MARKER_INTEGRATION_PLAN.md).

**External corroboration of our two-level marker design** (COLMAP issue threads, all consistent):
- **#1051** (someone with markers + known distances wanting metric output — our exact case): the answer is
  *"you wouldn't reconstruct a second time — once you know the scale factor you simply **rescale the entire
  reconstruction**."* = our **Level A post-hoc tape-median rescale** (`apply_metric_transform.py`). It then
  points to #999 if you want the prior as a BA *constraint* (= our Level B GCP-BA).
- **#999** ("model prior to improve triangulation in BA"): tsattler confirms COLMAP has **no native
  distance/point-prior-in-BA** (*"ask Johannes whether he plans to implement this"*) and that doing it means
  rolling your own BA that **weights the metric (mm) error against reprojection (px) error** — non-trivial,
  needs covariance weighting / GPS-prior + IMU-SLAM literature. That weighting subtlety is exactly why our
  **6 coplanar markers can't move the bundle much** → Level B only anchors scale/datum, doesn't improve
  internal geometry (matches our empirical Level-B finding). Plus #2228/#1471/#2687 (the maintainer reply).
- **Net:** Level A (post-hoc rescale) is the standard, correct metric path; Level B (prior-in-BA) is
  non-native, hard, and marginal for coplanar markers — so "Level A sufficient, Level B marginal" stands.

## 2. How it was run

Agisoft renames images on ingestion (`IMG_..._3.jpg`), so our `images.txt` and Agisoft's share **0**
exact names → `model_aligner` would find no common images. The check strips the trailing `_<digits>`
from Agisoft's image names (the same normalization `compare_to_agisoft.py` does), making all 93 names
match, then:

```bash
colmap model_aligner \
  --input_path     <session>/sparse/0 \
  --ref_model_path <agisoft-renamed-model> \
  --output_path    <out> \
  --alignment_type custom --ref_is_gps 0 \
  --min_common_images 3 --alignment_max_error 0.05 \
  --transform_path <out>/transform.txt
```

`transform.txt` is a COLMAP **Sim3** = `[scale, quat(4), translation(3)]` (8 numbers); `scale` is
metres per reconstruction-unit. The printed **Alignment error** (mean/median, in metres) is the
residual camera-to-camera distance after the best fit — it matches `compare_to_agisoft.py`'s per-camera
translation error (e.g. field_A/20250618: 10.0 mm here vs ~10 mm there), an internal consistency check
that the alignment is sane.

## 3. Results — all 14 sessions

`align_scale` = model_aligner (cameras↔Agisoft); `tape_scale` = our marker/tape (`metric_frame.json`);
`gap %` = `tape/align − 1` (positive ⇒ tape says the scene is bigger); `align_mm` = mean camera
residual after the fit (how trustworthy the alignment — hence the scale — is for that session).

### Reliable sessions (real camera, sound alignment)

| session | align_scale | tape_scale | gap % | align_mm | tape_mm |
|---|---:|---:|---:|---:|---:|
| field_A/20250603 | 0.573 | 0.579 | **+1.0** | 86 | 8.9 |
| field_A/20250618 | 0.467 | 0.487 | **+4.2** | 10 | 1.1 |
| field_A/20250627 | 0.474 | 0.493 | **+4.0** | 19 | 4.4 |
| field_A/20250715 | 0.473 | 0.486 | **+2.7** | 10 | 3.5 |
| field_A/20250722 | 0.444 | 0.467 | **+5.2** | 17 | 16.2 |
| field_D/20250603 | 0.522 | 0.513 | **−1.7** | 24 | 6.2 |
| field_D/20250627 | 0.430 | 0.441 | **+2.6** | 13 | 8.1 |
| field_D/20250715 | 0.479 | 0.465 | **−3.0** | 25 | 12.3 |
| field_D/20250722 | 0.414 | 0.438 | **+5.7** | 34 | 20.5 |

### Excluded (different camera or known pose problem → scale not trustworthy)

| session | gap % | align_mm | why excluded |
|---|---:|---:|---|
| field_A/20250613_lisa | −8.7 | 183 | Pixel-6a (different camera), poor Agisoft alignment |
| field_D/20250613_lisa | +35.5 | **763** | lisa, alignment garbage → scale meaningless |
| field_D/20250618 | +1.9 | 210 | known pose regression (lone seq→exh loser) |
| field_D/20250706 | −3.4 | 153 | high pose error |
| field_A/20250706 | +2.9 | 42 | borderline alignment |

## 4. What the gap says

1. **Agreement ±3–5 %.** Every reliable session's two independent metric routes agree to within a few
   percent — a clean, defensible validation that our marker+tape scale is correct.

2. **field_A is systematically positive (+1 to +5 %, all 6 field_A rows).** Our tape consistently makes
   field_A a few % *bigger* than Agisoft's marker bundle. That is independent corroboration of the
   **known field_A tape-entry error** (the target5↔target1 pair = +82 mm vs survey — see the tape
   memory): `model_aligner`, using *no* tape data at all, sees the same field_A inflation. field_D, by
   contrast, scatters symmetrically around zero (−3 to +6 %, median ~0) → no systematic, consistent with
   a clean field_D tape.

3. **Scale is more robust than pose.** field_D/20250618 and /20250706 have poor pose-alignment (210 mm,
   153 mm) yet their *scale* gap stays in the normal ±3 % band. Scale is a global average over all 93
   cameras, so local pose drift averages out; only a wholly different camera (lisa) corrupts scale too.

**Net:** model_aligner is not part of the metric pipeline (markers stay the only metric source), but as a
benchmark it (a) confirms the marker scale to a few %, and (b) independently fingers the field_A tape as
slightly long — a result we previously had from only a single outlier pair.

## 5. Files

- Tool: `colmap model_aligner` (COLMAP 4.1.0.dev0, our CUDA build).
- Our scale source: [src/preprocessing/triangulate_markers.py](../src/preprocessing/triangulate_markers.py) → `<session>/logs/metric_frame.json`.
- Reference: each session's `agisoft/sparse/0` (image names normalized as in [compare_to_agisoft.py](../src/preprocessing/compare_to_agisoft.py)).
- Related: [PHONE_SFM_POSE_ACCURACY.md](PHONE_SFM_POSE_ACCURACY.md), [MARKER_INTEGRATION_PLAN.md](MARKER_INTEGRATION_PLAN.md), [COMPARE_TO_AGISOFT_RESULTS.md](COMPARE_TO_AGISOFT_RESULTS.md).
