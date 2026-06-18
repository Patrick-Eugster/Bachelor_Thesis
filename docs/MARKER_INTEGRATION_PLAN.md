# Marker Integration into COLMAP — Plan & Data Map

Plan for bringing the surveyed **coded ground markers** into our COLMAP pipeline. Status: **planned, not yet coded.** Next concrete step is the **phone marker detector** (Stage A below).

---

## 1. Why — the goal

Our COLMAP currently has **no metric scale** (it relies on a Umeyama transform onto Agisoft to recover meters — scale ≈ 0.559 on `field_A/20250609`). The markers give us:

1. **A metric-accuracy number** — marker-to-marker distances from our reconstruction vs ground truth, directly comparable to the supervisor's Agisoft "Dist Error" (~5–15 mm). **This is the immediate target.**
2. **Metric scale independent of Agisoft** (a true phone-only pipeline) — later.
3. **Optional**: help SfM (tie-points) / refine calibration — later, low priority.

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

Everything except Step 1 already exists. **Detect on `images/`** (undistorted — matches our `sparse/0` poses). Outputs → `{source_path}/marker_vis/*.png` + `{source_path}/logs/marker_*.json`. Read-only on the data, no pipeline changes.

**Immediate next step — Stage A:** `detect_markers.py` doing **localization + overlay PNGs** on `field_A/20250609`, then **eyeball the overlays** to confirm all 6 targets are found before adding IDs (Stage B) or triangulation (Step 2).

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

---

*Related: [SFM_PIPELINE_COMPARISON.md](SFM_PIPELINE_COMPARISON.md) (the metric-scale + marker-GCP gaps), [COMPARE_TO_AGISOFT_RESULTS.md](COMPARE_TO_AGISOFT_RESULTS.md) (pose + intrinsics + point-cloud comparison), [AGISOFT_QUALITY_METRICS.md](AGISOFT_QUALITY_METRICS.md) (3D Error vs Dist Error definitions).*
