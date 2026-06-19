# Marker Integration into COLMAP — Plan & Data Map

Plan for bringing the surveyed **coded ground markers** into our COLMAP pipeline.

**Status (current):** Stage A (the marker *localizer*) is under active development — four heuristic
versions built (v1 square → v2 ellipses → v3 fiducial → v4 hybrid; full write-up in
[`MARKER_DETECTION_VERSIONS.md`](MARKER_DETECTION_VERSIONS.md)). Heuristics have **hit their ceiling**
(recall/precision/center can't all be satisfied by hand-tuned rules on the canopy). **Next step:
Option A = template-match the central bullseye (no training), escalating to Option B = a trained
CNN/YOLO detector if needed.** Detection is the **prerequisite for everything below** — and for both
the post-hoc *and* the in-SfM use of markers.

---

## 1. Why — the goal

Our COLMAP currently has **no metric scale** (it relies on a Umeyama transform onto Agisoft to recover meters — scale ≈ 0.559 on `field_A/20250609`). The markers give us:

1. **A metric-accuracy number** — marker-to-marker distances from our reconstruction vs ground truth, directly comparable to the supervisor's Agisoft "Dist Error" (~5–15 mm). **This is the immediate, easy validation target.**
2. **Markers used IN the SfM itself** — the real goal (decided with supervisor's framing), not just post-hoc. See §11b. Gives metric scale + better SfM natively.
3. **Optional**: refine calibration — later, low priority.

**Post-hoc vs in-SfM — both need the detector first.** Whether we triangulate markers *after* SfM (post-hoc, easy, validates the detector) or feed them *into* SfM (the goal, §11b), the prerequisite is identical: the 2D marker points from Stage A. So the detector is step one either way. Natural order: get detection solid → post-hoc validation → inject into SfM.

## 2. What the markers are

Agisoft **12-bit coded circular targets**: a solid **black central disk** with a **white center dot** (the precise point), ringed by **12 angular sectors** (black/white) that encode the ID. White square backing. **6 per phone field** (`target 1`–`6`), **3 per FIP plot**. Mounted on **stakes at canopy height** → visible in side-view phone images for both May and June sessions. Spec: `reference/agisoft/Coded_12bit_15cm-square_13cm-outer-circle_.pdf`.

## 3. The key concept — detection is only HALF the job

To get a marker's **3D position** you need TWO ingredients:

```
[1. 2D dots: where the marker is in each photo]  +  [2. camera poses (from SfM)]
              = a "marker DETECTOR"                          = we already have ours
                                  │
                                  ▼
        triangulate → 3D position → marker-to-marker distances → compare to truth = ACCURACY
                                  └──────────── the valuable part ────────────┘
```

A **detector only produces ingredient #1** (the 2D dots). All the value — 3D, distances, the accuracy number — comes *after*, and that downstream half is **identical** whether the 2D dots come from a CSV or from our own detector.

## 4. Data map — TWO separate experiments, don't mix them

| | **Phone** (demoanlage `field_A`–`D`) | **FIP** (plots 461–467, new folder) |
|---|---|---|
| **2D projections** (where a marker is in each photo) | ❌ none given → **we must DETECT** | ✅ `marker_projections.csv` (Agisoft gold) |
| **3D positions** (surveyed GT *locations*) | ✅ `metadata/markers/*_coordinates.txt` (+ `joaquin-*.csv`), CH1903+ metric | ❓ not in our data |
| **Distances** (tape-measured GT) | ✅ `metadata/markers/...manual-distances.xlsx` (cm) | ❓ not in our data |

- **Phone has the TRUTH** (positions + distances) **but no 2D dots** → we must build the detector. **← current focus.**
- **FIP has the 2D dots** (gold) **but no TRUTH** → **parked** until the supervisor provides FIP surveyed/manual distances.

`marker_projections.csv` columns: `Marker, Camera, X, Y, Pinned` — i.e. "marker *M* is at pixel (X,Y) in photo *Camera*"; `Pinned=True` = genuinely detected by Agisoft, `False` = back-projected. **Coordinates are w.r.t. the UNDISTORTED images** (use `cv2.undistortPoints` for the distorted variants).

## 5. What "distance" means

The **marker-to-marker** straight-line distance in meters (markers are physical plates in the field):

```
   target 1 •────────── 1.85 m ──────────• target 2
             \                           /
           2.40 m                    1.30 m
               \                       /
                •  target 3
```

6 markers → 15 pairs; 3 markers → 3 pairs. Distance is **transform/scale-invariant**, so comparing distances tests geometry+scale **without** any coordinate alignment.

## 6. Validation logic (answer-key analogy)

- **Given coordinates / tape distances = the answer key** (the truth).
- **Our pipeline** (detect → colmap → triangulate → distances) = a student solving from the photos alone; it never sees the key.
- **Compare = the accuracy score** (the thesis number).

We make the pipeline solve it even though we already have the key **because we are testing the pipeline, not looking up the answer**. Using the given coordinates directly would teach us nothing about our pipeline's accuracy.

## 7. Detector approach (decided)

**Identify by picking the best of the 6 KNOWN codes — not general decoding.**

| Option | Verdict |
|---|---|
| Full 12-bit decode / **CCTDecode** | Rejected as primary — overkill (we have only 6 known codes) + risky (must reproduce Agisoft's exact bit scheme; **fragile on oblique/occluded rings** — decoding needs the *whole* ring clean). Kept as optional later upgrade. |
| **Template-match the 6 PDF codes** ✅ | Robust, self-contained, inherits Agisoft's scheme via the rendered templates, works per-image (pre- *and* post-SfM), **degrades gracefully** under occlusion. **Chosen.** |
| **Geometric-ID** (match 3D layout to the 6 GT points) ✅ | Occlusion-immune (reads no ring), but only labels *after* triangulation. **Chosen as cross-check.** |

**Localization** (the risky part on a green canopy): black disk + white center-dot + ellipse fit + white-square-backing test → **sub-pixel center**. Disk-based ⇒ **immune to ring occlusion**.

**Not all 6 markers are visible in every image — and that's fine:** per-marker methods just label whatever is found; each marker only needs to appear in **≥2 images total** (guaranteed across ~90–120 views) to be triangulated.

## 8. Plan for PHONE (current focus)

```
STEP 1  Detect the 6 markers in each phone image        ← detect_markers.py (we build this)
STEP 2  Triangulate with our COLMAP poses (sparse/0)    → our 3D marker positions
STEP 3  Compute our marker-to-marker distances
STEP 4  Compare to GT distances (tape xlsx / survey)    → accuracy vs Agisoft's ~15 mm
```

Everything except Step 1 already exists. **Detect on `images/`** (undistorted — matches our `sparse/0` poses). Outputs → `{source_path}/marker_vis*/*.png` + `{source_path}/logs/marker_detections*.json`. Read-only on the data, no pipeline changes.

**Step 1 status:** four heuristic localizers built (v1–v4, see [`MARKER_DETECTION_VERSIONS.md`](MARKER_DETECTION_VERSIONS.md)); v4 (hybrid, contrast-relative thresholds) is best but still has poor recall + occasional FPs + center drift → **heuristics hit their ceiling**. **Immediate next step = Option A: template-match the central bullseye** (rotation-invariant; correlation peak = exact center; no training). If recall on distant/occluded plates is insufficient → **Option B: a trained CNN/YOLO marker detector** (repurpose the repo's YOLO infra; label 6 markers in ~30–40 images; fiducial-snap inside each box for sub-pixel center). Only after Step 1 is solid do we move to Stage B (IDs) + Step 2 (triangulate).

## 9. Where markers actually help (honest)

- **Metric scale + validation** — the big win; 6 markers are plenty.
- **Tie-points / helping SfM** — marginal on already-working sessions (6 points vs ~50k SIFT tie-points); only real upside on repetitive/marginal sessions (e.g. blurry `20250530`).
- **Calibration** — marker projections refine intrinsics, *caveat:* our markers are roughly coplanar → weak for lens **distortion**, fine for focal/scale.
- Agisoft's own metric step (`importReference` + `updateTransform`, [reference/agisoft/6-…py](../reference/agisoft/6-agisoft_preprocessing_demoanlage_2025.py)) is essentially a **marker-datum similarity transform** — conceptually the same as our post-hoc Procrustes.

## 10. Caveats

- **Stakes were raised** over the season → absolute Z vs the March survey is unreliable → **horizontal/XY distances are the safe GT**, absolute georeferencing is shaky.
- **No dense cloud exists** (ours or Agisoft) → any dense comparison must first *generate* one (`patch_match_stereo` + `stereo_fusion`).
- **FIP needs GT distances** from the supervisor before it's usable for validation.
- GT distances live in the **xlsx** (needs `openpyxl`) — or derive them from the surveyed positions (dependency-free).

## 11. FIP gold data (parked, available)

`demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed/`: plots 461–467, each with `colmap_reprocessed/` in **4 variants** (`{distorted,undistorted} × {jpg,png}`) + `marker_projections.csv`. The **distorted originals** would also let us run *our own* COLMAP on FIP (previously FIP came only as Agisoft output). Parked until FIP GT distances arrive; meanwhile it's an optional **sanity-check of the triangulation math** on gold 2D dots.

## 11b. Using markers IN the SfM (the real goal, not post-hoc)

Post-hoc triangulation only *measures* accuracy. The actual aim is to feed markers **into** COLMAP. Two levels (both need Stage A detections first):

- **Level A — markers as reliable tie-points.** Inject the detected 2D marker points into COLMAP's database as extra cross-image matches. COLMAP triangulates + bundle-adjusts them with the SIFT points. Because markers are unambiguous and repeatable (unlike SIFT on **repetitive wheat** — the reason we already need `single_camera` + exhaustive matching), they add strong correct constraints → **better connectivity, less drift**. This mirrors Agisoft's `detectMarkers` *before* alignment.
- **Level B — markers as Ground Control Points (GCPs).** Attach the **known surveyed XYZ** to the marker points and constrain the reconstruction to them → the model comes out in **real metric scale**, anchored to the markers, **no Umeyama-to-Agisoft needed**. In COLMAP: `model_aligner` with control points, or a constrained bundle adjustment via `pycolmap`.

Full Agisoft-equivalent = **A + B**. More engineering (write marker observations into the COLMAP DB / use pycolmap), but very doable, and it's the genuine thesis contribution.

## 11c. How false positives / detection quality actually matter (clarifications)

- **Stage A only *proposes*.** "Fewer false positives in vX" = fewer *eyeballed* wrong proposals vs the image, not geometry-confirmed. The **rigorous** false-positive rejection is downstream: **multi-view triangulation consistency** (a real marker's rays intersect; a canopy FP's don't → rejected) + **ID matching** (Stage B: a real marker matches one of the 6 codes; an FP matches none).
- **Coverage that matters = per-marker, not per-image.** Each of the 6 markers needs to be localized in **≥2 views over the whole set** to triangulate. Per-image count is irrelevant. More views per marker → better (noise averaging + RANSAC outlier rejection), diminishing after ~5–10.
- **FPs do NOT touch COLMAP/3DGS in the post-hoc plan** — those are already trained without markers; a stray FP only adds noise to the marker triangulation (and usually won't triangulate at all). **Only if markers are injected into SfM (§11b) would an FP bias the reconstruction** — which is why detection precision + the ID/geometry filters matter before doing Level A/B.

---

*Related: [SFM_PIPELINE_COMPARISON.md](SFM_PIPELINE_COMPARISON.md) (the metric-scale + marker-GCP gaps), [COMPARE_TO_AGISOFT_RESULTS.md](COMPARE_TO_AGISOFT_RESULTS.md) (pose + intrinsics + point-cloud comparison), [AGISOFT_QUALITY_METRICS.md](AGISOFT_QUALITY_METRICS.md) (3D Error vs Dist Error definitions).*
