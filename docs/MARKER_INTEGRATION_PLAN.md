# Marker Integration into COLMAP — Plan & Data Map

Plan for bringing the surveyed **coded ground markers** into our COLMAP pipeline.

**Status (current): Stage A + B SOLVED via Option C = CCTDecode (detect-and-decode).** Full write-up in
[`MARKER_DETECTION_CCT.md`](MARKER_DETECTION_CCT.md); version history in
[`MARKER_DETECTION_VERSIONS.md`](MARKER_DETECTION_VERSIONS.md).
- **v1–v6** (heuristics → template matching) only *localised* and hit a ceiling (false positives, no IDs).
- **Pivot to Option C — decode the 12-bit code.** The code self-validates (wheat can't form a valid
  codeword → false positives vanish) **and gives the ID for free** (= the old "Stage B"). The earlier worry
  that decoding is "fragile on oblique/occluded rings" was resolved: **forcing the decode onto the central
  disk** (instead of CCTDecode's own blob search, which grabbed arcs) makes all 6 markers decode to distinct,
  consistent IDs.
- **v7** (`detect_markers_v7_cct.py`): v6 proposes centers → forced-center decode; **fill-ratio** disk-vs-arc
  discriminator (size-independent); disk-center reported (0.7 px). **73% recall of Agisoft.**
- **v8** (`detect_markers_v8_cct.py`): concentric-consensus + re-centering — recovers the center from the
  arcs when v6 lands on an arc or the disk is occluded. **76% recall**; fixes the arc-vs-disk ID flips.
- **GT validation in place:** Agisoft `marker_projections.csv` (phone) + `compare_v7_vs_agisoft.py` →
  recall, per-target ID map (113↔T1 … 77↔T6), miss CSVs. Remaining misses are decode-resolution on far/small
  markers (per-marker we have 10–27 views each — complete).

**Next:** min-views cleanup (drop the noise-tail extras) and/or **triangulation** (vote IDs, get 6 3D
markers), then feed `(marker_id → pixel xy)` + 3D as **GCPs** into COLMAP (§11b / Option D — now native in
COLMAP). Detection is the **prerequisite for everything below**, and it's now done. (Option B trained
CNN/YOLO kept in reserve, not needed.)

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

## 7. Detector approach (REVISED — Option C / CCTDecode won)

**The plan originally rejected full decoding as "fragile on oblique/occluded rings" and chose
template-matching. That call was reversed** — see [`MARKER_DETECTION_CCT.md`](MARKER_DETECTION_CCT.md):

| Option | Verdict (updated) |
|---|---|
| **CCTDecode 12-bit decode** ✅ | **ADOPTED.** The fragility worry was about CCTDecode's *own blob search* grabbing arcs — not the decode itself. **Forcing the decode onto the central disk** (v7/v8) makes all 6 markers decode to distinct, consistent IDs. The decode **self-validates** (kills false positives) and **gives the ID for free**. We don't reproduce Agisoft's bit scheme — IDs are **consistent-only** (a 6-row hand map covers GT lookup). |
| Template-match the 6 PDF codes | Superseded — v5/v6 template matching only localised and hit a ceiling. |
| **Geometric-ID** (match 3D layout to GT) | Still useful as a **post-triangulation cross-check** of the decoded IDs. |

**Localization** (the risky part on a green canopy): the **fill ratio** (blob area ÷ fitted-ellipse area)
separates the solid **disk** (~1.0) from a code **arc** (≤0.91), size-independently; v8's concentric
reconstruction recovers the center from the **arcs** when the disk is occluded ⇒ **robust to occlusion of
either the disk or some arcs**.

**Not all 6 markers are visible in every image — and that's fine:** per-marker methods just label whatever is found; each marker only needs to appear in **≥2 images total** (guaranteed across ~90–120 views) to be triangulated.

## 8. Plan for PHONE (current focus)

```
STEP 1  Detect the 6 markers in each phone image        ← detect_markers.py (we build this)
STEP 2  Triangulate with our COLMAP poses (sparse/0)    → our 3D marker positions
STEP 3  Compute our marker-to-marker distances
STEP 4  Compare to GT distances (tape xlsx / survey)    → accuracy vs Agisoft's ~15 mm
```

Everything except Step 1 already exists. **Detect on `images/`** (undistorted — matches our `sparse/0` poses). Outputs → `{source_path}/marker_vis*/*.png` + `{source_path}/logs/marker_detections*.json`. Read-only on the data, no pipeline changes.

**Step 1 status: DONE (Stage A localize + Stage B ID together) via Option C / CCTDecode** — see
[`MARKER_DETECTION_CCT.md`](MARKER_DETECTION_CCT.md). v1–v6 (heuristics → template matching, helper
`make_fiducial_template.py`) only localised and hit a ceiling; we pivoted to **decoding the 12-bit code**.
**v7** (`detect_markers_v7_cct.py`) = v6 proposes centers → forced-center CCTDecode (fill-ratio disk finder,
disk-center 0.7 px) → **73% recall of Agisoft**; **v8** (`detect_markers_v8_cct.py`) = + concentric-consensus
+ re-centering (recovers center from arcs when v6 lands on an arc / disk occluded) → **76% recall**, fixes the
arc-vs-disk ID flips. Decode gives the **ID for free** (Stage B done) and self-validates. Validated with
`compare_v7_vs_agisoft.py` against Agisoft `marker_projections.csv`. **Next = Step 2 (triangulate + vote IDs),
then GCPs into COLMAP (§11b).** (Remaining per-view misses are decode-resolution on far/small markers;
per-marker coverage is 10–27 views each = complete.)

**v8 + view-filter + cross-session (4 phone sessions).** Added a cross-image **view filter** to v8
(`keep_top_k = expected_markers` IDs above a `min_views` floor; dropped detections kept in
`per_image_dropped` with locations). v8-filtered beats v7 on both axes (118 vs 113 real obs, 0 vs 12 junk
IDs on 20250609). Run on all 4 sessions — recall of Agisoft (localization, tol 25px): `field_A/20250609`
**76%**, `field_A/20250618` **89%**, `field_D/20250523` **85%**, `field_D/20250530` **1%** (blurry — 3× less
sharp, 22 imgs unregistered → **unusable for markers, excluded**). The `keep_top_k` heuristic proved fragile
(pulled a junk ID into the top-6 on 20250618/20250523), which motivated the **code-structure findings** →
see **[`MARKER_CODE_STRUCTURE.md`](MARKER_CODE_STRUCTURE.md)**: legal set = 352 of 4096, `B2I`
rotation-canonicalization, **legal ≠ separated** (min Hamming 1). **Manifest = ground truth decoded from the
spec PDF** (6 pages → target1=113 … target5=**85** … target6=77) → our 6 deployed codes
`{77,85,89,101,105,113}`, all legal + mutually ≥2; **117 = 1-bit misread of 85 (target 5)** confirmed.
Filter built: a **necklace "dictionary" does NOT filter junk** (decoded codes always canonical → always
legal), so the real filter is the **plot manifest** — implemented as v8 `id_filter=manifest` (the default,
PDF codes; dropped dets kept with locations in `per_image_dropped`). It cleanly drops both wheat-junk and
near-neighbour misreads (117, 1535) → all 3 good sessions now keep exactly the 6 real codes. Re-run to
`marker_vis_v8_manifest/` + `..._v8_manifest.json`; recall vs Agisoft 76/86/84% (20250618 89→86% is honest —
the old number counted 117's 15 detections as location-hits; manifest keeps only the true 85). Final
disambiguation (snap 117→85) is the **triangulation-by-location** step — majority vote / Hamming alone unsafe.

**STEP 2 DONE — triangulation (`src/preprocessing/triangulate_markers.py` + yaml).** Modular standalone
(reads any detector JSON + `sparse/0/`; reuses `colmap_loader` + scipy). Per marker: DLT + RANSAC + LM
refine → 3D point; **snap** dropped near-neighbour misreads back by LOCATION (Hamming-guarded); **seed** an
under-covered marker from its misreads (e.g. 85 rebuilt from the 117s — the chicken-and-egg fix);
**reproject** each 3D point into every frame to recover missed/glared views. Outputs `logs/marker_points3d.json`
(6 pts + reproj err + parallax) + `logs/marker_triangulation.json` (per-obs detected/snapped/reprojected) +
`marker_vis_v8manifest_triangulated/`. **Result: all 3 good sessions solve 6/6 markers**, median reproj 0.5–3.1 px,
parallax 37–112°. field_A/20250618 target5=85 recovered from 117s (15 views, 1.13px). **Validated vs Agisoft
(20250609): detected pos 0.7px, reprojected pos 2.3px, and ALL 127 Agisoft GT observations now covered (0
missed) — Agisoft's `Pinned` reprojection reproduced.** Caveat: reproject counts in-bounds+cheirality only
(not occlusion) → the ~270–590 "reprojected" per session are CANDIDATE positions, not all truly visible; the
GT-validated subset is accurate to ~2.3px.

**We MATCH Agisoft fully and BEAT it slightly on detection recall (20250609):** of our 118 detected
manifest observations, 96 coincide with an Agisoft pin and **22 are EXTRA** (real markers Agisoft chose not to
pin — Agisoft is deliberately low-recall; the extras decode to manifest codes AND are triangulation inliers,
so they're genuine). The 31 Agisoft pins we didn't detect are recovered by reprojection → 127/127. So
defensible claim = **all 127 covered + 22 genuine extras**; the ~590 reprojections are an unvalidated superset,
not a "found more" claim.

**Snap/seed = PURE GEOMETRY (Hamming guard dropped, `snap_hamming_max=0`):** a misread can flip >1 bit under
occlusion, so a Hamming≤1 guard wrongly excludes real recoveries; assignment is by 3D-reprojection location +
RANSAC, code-agnostic. Pure-geometry recovered MORE (snapped 6/18/10 vs the guarded 4/11/5), all 6/6 still
solved.

**STEP 3 DONE — metric scale (`src/preprocessing/marker_scale.py` + yaml).** Recovers the COLMAP→metres scale
from the 6 triangulated points against TWO independent references: **PRIMARY = surveyed XYZ**
(`demoanlage2025_v0/metadata/markers/field_<L>_coordinates.txt`, CH1903+/LV95 metres — dependency-free, plain
text; target→code map in §5 of MARKER_CODE_STRUCTURE.md), **CHECK = tape-measure xlsx**
(`Demoanlage-2025-markers-manual-distances.xlsx`, sheet `plot <L>`, an upper-triangular 6×6 cm matrix — parsed
straight from the zipped XML, **no openpyxl needed**). Computes scale two ways (median distance-ratio + rigorous
**Umeyama** similarity fit) and reports a per-marker mm residual after alignment + a survey-vs-tape-vs-ours
per-pair table. Output `logs/marker_scale.json`.

**Results (all 3 good sessions):**

| session | scale (m/unit) | ratio CV | Umeyama RMS | ours vs survey (15 pairs) | tape vs survey |
|---|---|---|---|---|---|
| field_A/20250609 | 0.5590 | 1.37% | 19.3 mm | 13.8 mm | 17.1 mm |
| field_A/20250618 | 0.4773 | 1.51% | 21.0 mm | 15.2 mm | 17.1 mm |
| field_D/20250523 | 0.4627 | 1.46% | 18.1 mm | 15.3 mm | 8.2 mm |

Per-session scales differ (each COLMAP run has its own arbitrary unit — expected). **Headline: phone-COLMAP
markers recover metric scale to ~14–15 mm distance accuracy / ~18–21 mm RMS after alignment — comparable to
Agisoft's ~5–15 mm.** The two refs agree: the ratio CV ≈ 1.4–1.5% everywhere = rigid geometry (good
triangulation, no warp). On field_A our distances actually match the survey *better* (13.8 mm) than the **tape**
does (17.1 mm) → the hand tape is the coarser reference, not us (its one gross outlier, pair (85,113) at +82 mm,
is a tape entry error; our value is +34 mm). The per-field survey/sheet selection is verified correct: field_D
distances differ from field_A (e.g. (77,85) 0.94 m vs 0.73 m).

**LEVEL B / FLAVOUR 1 DONE — metric model (`src/preprocessing/apply_metric_transform.py` + `marker_metric.yaml`).**
Applies the Step-3 similarity transform to the whole `sparse/0/` model → a metric `sparse_metric/` (real
metres). Geometry unchanged (rigid+scale, no re-optimisation); the point is metric units for **phenotyping**
(length/width/volume in mm) + a native survey self-check. **GOTCHA (recorded): `colmap model_transformer`'s
`--transform_path` did NOT apply a plain `[sR|t]` 4×4** — it produced a scrambled rotation + 30 m translation
(caught by a built-in camera-centre convention check; 3×4 segfaults, 4×4 misparses). So we **rewrite the COLMAP
TEXT model in Python** instead: transform only the numeric coords (poses in `images.txt` AND COLMAP-4.1's
`frames.txt` RIG_FROM_WORLD; XYZ in `points3D.txt`), copy tracks/2D-obs/IDs/`cameras`/`rigs` verbatim, text-only
output (no stale `.bin`). Pose map under world similarity X'=sRX+t: `R_wc'=R_wc Rᵀ`, `t_wc'=s·t_wc − R_wc Rᵀ t`
(⇒ camera centre C'=sRC+t, verified to 1e-11 mm). **LOCAL origin** = survey centroid (coords near 0) so 3DGS
float32 stays precise — would be ~0.25 m ulp at CH1903's 2.69e6 m; the CH1903 origin is saved in
`logs/metric_frame.json` for georeferencing. **Result (3 sessions): valid metric models, pose write-back exact,
marker RMS 19.3/21.0/18.1 mm; sanity extents real** (field_A/20250609 cameras 4.3×4.1×0.24 m handheld
walk-around, cloud 12.9×10.7×1.0 m). Feeds 3DGS as a text model (FIP path already reads text).

**LEVEL B / FLAVOUR 2 DONE — GCP-constrained BA (`src/preprocessing/marker_gcp_ba.py` + `marker_gcp_ba.yaml`,
via pycolmap).** The COLMAP CLI exposes NO marker-GCP BA (only `pose_prior_mapper` = camera-GPS priors;
`model_aligner` = a 7-DOF similarity = Flavour 1), and pycolmap 4.0.4 has no dedicated GCP class — but its
general `BundleAdjuster` + `BundleAdjustmentConfig` build it: load the Flavour-1 metric model, add the 6 markers
as 3D points AT THE SURVEY positions with their real 2D obs (detected/snapped inliers, NOT reprojected),
`add_constant_point` them (= the GCP), `add_variable_point` every scene point, `add_image` all, run BA → poses +
scene re-optimise to honour the survey. Intrinsics fixed (scale is pinned by the GCPs). (`pip install pycolmap`,
4.0.4 — standalone wheel, torch untouched; `image.points2D` supports in-place `.append()` though whole-list set
is locked.) Output: refined `sparse_metric_gcp/` (binary) + `logs/metric_gcp_ba.json`.

**RESULTS — two robust facts + a revealing split:**

| session | scene reproj (px) | marker reproj px (before→after) | cam shift | reads as |
|---|---|---|---|---|
| field_A/20250609 | 1.24 → 1.24 | 30.7 → **18.6** | 23 mm | residual STAYS |
| field_A/20250618 | 1.32 → 1.32 | 37.4 → **22.7** | 28 mm | residual STAYS |
| field_D/20250523 | 1.29 → 1.29 | 26.4 → **3.8**  | 28 mm | residual ABSORBED |

(1) **Scene reprojection is byte-identical before/after on every session** — anchoring never harmed the
reconstruction. (2) BA always reduced marker reprojection by shifting cameras ~23–28 mm. **The split lines up
with the earlier tape-vs-survey agreement:** field_D (tape agreed with survey to 8 mm) → BA drives markers to
**3.8 px ≈ scene level** = survey FULLY consistent with imagery → here GCP-BA genuinely improves on Flavour 1.
field_A (tape disagreed 17 mm) → markers plateau at ~18–23 px ≫ the 1.3 px scene = a REAL, irreducible
survey↔imagery disagreement BA can't absorb without distorting the self-consistent scene (it refused). At
~2.5 m phone range 18 px ≈ 18 mm, matching the Flavour-1 residual.

**CONCLUSION (honest):** GCP integration WORKS (field_D is the proof); where a residual remains (field_A) it is
**survey/data-limited, not a pipeline fault** — the scene stays internally consistent at 1.3 px throughout. So on
field_A the ~18 mm is most likely the SURVEY's own RTK-GPS-level error (or a small marker-detection bias), not
our phone reconstruction. **PROVISIONAL (3 sessions only — field_A/20250609+20250618, field_D/20250523):**
Flavour 1 (post-hoc similarity) *looks* sufficient for accuracy on these, and Flavour 2 didn't beat it; but this
is NOT a final verdict — all three happened to agree, and Flavour 2 (a properly bundle-adjusted metric model +
the diagnostic that pinpoints where the residual lives) could still earn its keep on a session where survey and
imagery disagree more, or where COLMAP's scale drifts. **Re-evaluate when more phone dates are added.** Matches
the prediction that 6 near-coplanar markers anchor scale/datum but can't improve internal geometry. **OPEN Q for supervisor:
which instrument produced `field_<L>_coordinates.txt` (RTK-GPS ~cm vs total-station ~mm)? — it decides whether
the field_A ~18 mm is survey-limited (likely) or reconstruction-limited.** Both metric models (`sparse_metric/`
Flavour 1, `sparse_metric_gcp/` Flavour 2) are available to feed 3DGS for metric phenotyping.

**ORCHESTRATOR-WIRED (`run_preprocessing.py`):** the whole marker layer is now steps 4-8 of the preprocessing
orchestrator, behind a master `run_markers` toggle (default OFF → base SfM pipeline unchanged) plus per-step
sub-toggles (`run_marker_detect/triangulate/scale/metric/gcp`). Run with
`python src/preprocessing/run_preprocessing.py field=field_A plot=20250609 run_markers=true`. Two safety nets:
(1) every marker step is **fail-soft** — a crash (missing survey file, local-only pycolmap) warns and continues
instead of aborting; (2) a **failsafe** counts solved 3D markers after triangulation and **skips the metric steps
6-8 if fewer than `min_markers` (default 4)** solved, so we never anchor metric size on 1-2 markers — the model
just stays in relative scale. The orchestrator prints a `MARKER LAYER SUMMARY` recap + a per-step TIMING table.
**Measured timing (field_A/20250609, markers-only):** detect 2:16 / triangulate 0:17 / scale+Flavour 1+Flavour 2
≈ 1 s total → **CCT detection is ~88% of the cost; the only step worth optimising.**

**TODO (future, low priority):** a coded-target **generator tool** — given the 352 legal 12-bit codes, return
K codes with **max-min Hamming ≥ 3** so future users deploy markers whose single-bit misreads are *uniquely*
correctable (our current markers are only min-distance 2). Forward-looking only; can't re-place existing
markers. Details in [`MARKER_CODE_STRUCTURE.md`](MARKER_CODE_STRUCTURE.md) §7.

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
