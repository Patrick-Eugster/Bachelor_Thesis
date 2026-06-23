# Experiment — does integrating markers *into* COLMAP beat the post-hoc approach?

**Branch:** `markers`  ·  **Status:** ARM A DONE (3 sessions; more dates to come)  ·  **Scope (first pass):** Arm A + leave-one-marker-out, geometry metrics only (3DGS deferred to Euler).

## RESULTS — first pass (`marker_gcp_lomo.py`, 6 folds/session)

| session | baseline_noBA (post-hoc) | frozen | focal | **focal_pp** | anchoring vs post-hoc |
|---|---|---|---|---|---|
| field_A/20250609 | **16.9 mm** | 37.6 | 37.6 | 37.7 | **HURTS** (+20.8) |
| field_A/20250618 | **19.9 mm** | 37.9 | 38.1 | 36.7 | **HURTS** (+16.8) |
| field_D/20250523 | 17.2 mm | 13.8 | 13.6 | **10.7 mm** | **HELPS** (−6.5) |

(numbers = mean held-out marker error in mm; lower is better. Scene reproj was **byte-identical** across
frozen/focal/focal_pp in every session — gauge freedom absorbs intrinsic changes into poses, so only the
independent held-out marker distinguishes the methods. That is the whole point of the LOMO design.)

**VERDICT — survey-dependent, and it matches the earlier Flavour-2 split exactly:**
- **Where the survey AGREES with the imagery (field_D)**: feeding markers into the BA *and refining
  intrinsics* (focal_pp) gives the best result by a clear margin — **10.7 mm, −38 % vs post-hoc**. Here
  in-BA integration genuinely improves the camera model.
- **Where the survey has its own error (field_A, the ~15 mm we suspect is RTK-GPS-limited)**:
  hard-anchoring those markers **bends the cameras to fit wrong targets** and *degrades* the independent
  held-out marker by ~17–21 mm. Worse, not better.
- **Intrinsic refinement alone is inert on bad-survey sessions** (scene reproj unchanged, held-out
  unchanged) — 6 coplanar markers don't independently constrain focal/cx/cy there.

**So:** Flavour 1 (post-hoc similarity) stays the **safe default** — it never degrades, it just scales the
self-consistent model. But there is **real upside** from intrinsic-refined GCP-BA on good-survey
sessions. The deciding factor is survey quality, which makes the open question — *which instrument
produced `field_<L>_coordinates.txt` (RTK-GPS ~cm vs total-station ~mm)* — directly actionable.

**Caveat / next lever (REVISED — soft GCPs demoted by the tape gate):** we use **HARD-constant** GCPs
(pycolmap `add_constant_point`), the pessimistic bound — it trusts the survey absolutely. A
**soft/weighted GCP** would keep field_D's gain *without* field_A's degradation. BUT we realised the
**tape gives a cheaper fix**: it tells us *in advance* whether a field's survey is good or bad (Step-3
tape↔survey cross-check, no re-run needed — see `MARKER_INTEGRATION_PLAN.md` §1b), so we can simply
**gate**: good survey → run focal_pp GCP-BA; bad survey → stay post-hoc (Flavour 1). That binary gate
handles our current cleanly-good (field_D) vs cleanly-bad (field_A) fields. **Soft GCPs are therefore
LATER / IF-NEEDED** — only for *mixed-quality within a field* (some good + some bad markers) or
*no-tape* sessions, where a per-marker weighting beats an all-or-nothing gate. Re-run this as more phone
dates land. Not escalating to Arm B/C — the hard-anchoring result already shows the ceiling.

## The question
Today the markers are applied **after** reconstruction:
- **Flavour 1** (`apply_metric_transform.py`) = a 7-DOF similarity fit on the finished model.
- **Flavour 2** (`marker_gcp_ba.py`) = one GCP-constrained BA pass with **intrinsics frozen**
  (`refine_focal_length/principal_point/extra_params=false`) and markers as CONSTANT points.

Measured result: scene reprojection stayed **byte-identical** before/after → marker anchoring changed
nothing about the internal geometry. **Does anchoring markers *during* optimization — letting them pull
on camera poses AND intrinsics — actually improve accuracy, or does it just confirm the current ~18 mm
residual is data-limited (survey error), not a pipeline fault?**

**Hypothesis (to confirm or break, NOT assume):** 6 near-coplanar ground markers are too weak a
constraint to improve COLMAP's already-good internal geometry (thousands of natural feature points), so
in-BA integration won't beat post-hoc. The experiment exists to test this, not to rubber-stamp it.

## The circularity trap (most important design point)
Anchoring on all 6 markers and then measuring marker error trains and tests on the same points — error
drops by construction, proving nothing. The honest test is **leave-one-marker-out (LOMO)**:

> Anchor on **5** markers, predict the **held-out 6th**'s metric position, measure its error (mm) vs the
> survey value. Rotate through all 6 holdouts → 6 held-out errors per method. *That* number says whether
> marker integration genuinely improves the model or merely fits the anchors.

Second, independent yardstick: **Agisoft** (`compare_to_agisoft.py`). Agisoft never saw our markers, so
per-camera translation/rotation error vs Agisoft is a clean external reference. All 3 candidate sessions
have `agisoft/sparse/0/`.

## Arms (this pass = A only; B/C are escalation)
| arm | what | effort | tooling |
|---|---|---|---|
| **Control** | current post-hoc: Flavour 1 similarity + Flavour 2 GCP-BA (intrinsics frozen) | done | — |
| **A** | GCP-BA with **intrinsics refined** (`refine_focal_length`, then `+principal_point`) — let markers correct calibration in the final BA | **low** (flags + LOMO harness) | pycolmap BA |
| B | global BA from full model + markers as GCPs + re-triangulate scene points | medium | pycolmap BA + triangulator |
| C | markers in the **incremental mapper from the start** (metric-from-scratch) | high | `pycolmap.IncrementalMapper` (reachable, fiddly) |

**Run A first.** It's a few config flags + the LOMO harness and directly answers "can markers fix
calibration." Escalate to B/C **only** if A shows a real signal; if even A can't move the needle, the
weak-constraint hypothesis holds and we stop.

## Datasets
The 3 good sessions (survey + Agisoft present): **field_A/20250609** (primary), **field_A/20250618**,
**field_D/20250523**. field_D is the most likely to show a gain — its tape agreed with survey and
Flavour 2 already *absorbed* the residual there (marker reproj 26.4 → 3.8 px), unlike field_A where it
stayed (~18 px).

## Metrics (per arm, per session)
1. **LOMO held-out marker error (mm)** — headline, circularity-free.
2. **Marker reproj (px)** before/after — anchor fit.
3. **Scene reproj (px)** before/after — did refining intrinsics HURT internal consistency?
4. **vs-Agisoft** camera translation (mm) + rotation (deg) — independent reference.
5. **Intrinsics drift** — how far focal/cx/cy moved when freed (large drift on 6 coplanar points = overfit/wobble).
6. *(deferred)* downstream **3DGS PSNR/SSIM/LPIPS** — only if a geometry win appears worth confirming.

## Decision rule
- **A wins** if LOMO error drops AND scene reproj + vs-Agisoft don't worsen → markers genuinely help;
  promote intrinsic-refinement as default, consider B/C.
- **A null** if LOMO flat/worse OR scene reproj rises (wobble) → confirms weak-constraint hypothesis;
  **Flavour 1 stays the recommendation**, document and stop.

## Implementation outline (Arm A)
- Extend `marker_gcp_ba.py` with a **LOMO mode**: loop over the 6 markers, each fold adds 5 as constant
  GCPs + refines (poses + intrinsics + scene points), then measures the held-out marker's 3D error (mm)
  by reprojection / triangulation against its survey value.
- Reuse the existing before/after reproj + camera-shift instrumentation; add intrinsic-drift logging.
- Add config flags (already present in `marker_gcp_ba.yaml`): flip `refine_focal_length` (then
  `+refine_principal_point`) for the Arm-A folds; keep a frozen-intrinsics control fold = current Flavour 2.
- Output: `logs/metric_gcp_lomo.json` + a results table in this doc across the 3 sessions → verdict.

## Notes / constraints
- All local (pycolmap 4.0.4, no Euler), gitignored data, reversible. pycolmap is local-only (NOT in
  pyproject — keep it that way to avoid the Euler torch-2.1.2 break).
- `image.points2D` is append-only in pycolmap (whole-list reassign is locked) — the LOMO folds must
  rebuild observations the same in-place way `marker_gcp_ba.py` already does.

See also: [`MARKER_INTEGRATION_PLAN.md`](MARKER_INTEGRATION_PLAN.md) (the full marker pipeline),
[`AGISOFT_QUALITY_METRICS.md`](AGISOFT_QUALITY_METRICS.md) (what the error numbers mean).

---

## Appendix — plain-language primer (the concepts behind this experiment)

**What "survey" is.** The surveyed real-world coordinates of each marker — someone went into the field
with a surveying instrument and recorded each marker's exact real-world position in metres
(`field_<L>_coordinates.txt`, Swiss CH1903+/LV95). It's like a GPS location but more precise. The
accuracy depends on the instrument: **RTK-GPS ≈ centimetres** (~10–20 mm error) vs **total-station ≈
millimetres**. We don't know which was used — that is the open question for the supervisor, and it is
exactly what decides whether a field's survey is "good." **Survey is per FIELD, not per session** (the
markers are staked in the ground and don't move), so `field_A/20250609` and `field_A/20250618` share one
survey — which is why they behave identically in the results above.

**How "error" is computed.** It's a 3D distance: `‖ our_predicted_marker_position − surveyed_position ‖`,
in mm. Both are 3D points in the same metric frame. "10.7 mm" = our model places the marker 10.7 mm from
its surveyed truth. Lower = more accurate.

**How LOMO (leave-one-marker-out) works.** Leaving a marker "out" does NOT mean forgetting it — we still
see its pixel in ~20 photos. It means we don't hand the bundle adjustment its survey coordinate as an
anchor. We then **predict** its 3D position by triangulation: shoot a ray from each camera through that
marker's pixel; the rays cross at one 3D point. Because that prediction never used the marker's survey
value, comparing it to the survey is a fair test. (Anchoring the marker AND then measuring it would just
force the answer — proving nothing. That's the circularity LOMO avoids.)

**What COLMAP itself computes (and its own quality).** The markers are an extra check we added — NOT part
of COLMAP's core. COLMAP computes (1) **camera intrinsics** = the lens calibration (focal length, optical
centre cx/cy, ±distortion); (2) **camera extrinsics** = each photo's position + orientation; (3) the
**sparse 3D point cloud** = thousands of natural feature points. COLMAP's native quality number is the
**scene reprojection error** (project those points back into the photos, measure the pixel gap; our
~1.2 px = good), plus how many images registered (93/93). **But reprojection only measures internal
self-consistency** — "do the photos agree with each other?" — and says nothing about real-world
correctness: a model can have 1 px reproj while being the wrong SIZE or slightly WARPED. The markers +
survey measure the thing reproj can't — **absolute metric accuracy** — which is why the held-out marker
error is a different, complementary number. (Calibration is also cross-checked vs Agisoft: focal −1.94%.)

**Does Agisoft do the same?** Yes — Agisoft Metashape is a full photogrammetry pipeline like COLMAP
(feature match → triangulate → bundle adjust → calibrate), but it **natively** integrates the coded
markers + survey control points to come out metric. Replicating that in open-source COLMAP is the point
of this whole marker effort; Agisoft's `marker_projections.csv` is its own marker triangulation, used as
our ground truth.
