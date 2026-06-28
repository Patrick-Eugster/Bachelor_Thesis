# Marker Integration into COLMAP — Plan & Data Map

Plan for bringing the surveyed **coded ground markers** into our COLMAP pipeline.

---

> ## 🎯 DECISION (2026-06-26): TAPE ONLY — survey XYZ is NOT used for now
>
> Metric scale comes **only from the hand-measured tape distances**. The surveyed XYZ
> (`field_<L>_coordinates.txt`) is **off the table for now** — it's RTK-GPS-limited (~2 cm,
> field_A tape↔survey gap 17 mm) and anchoring it bends good geometry to noisy targets (see
> `MARKER_COLMAP_RERUN_EXPERIMENT.md`). Tape is mm-level, needs no GPS, and is what any farmer has.
>
> **⚠️ KNOWN TAPE DATA ERROR (field_A / sheet "plot A"):** the tape distance for pair **target5↔target1
> (codes 85↔113) = 1.4580 m is a gross outlier** (~+82 mm vs survey 1.3756 m, +45 mm vs our
> photogrammetry ~1.40 m). Survey + our photogrammetry (two independent methods) agree ~1.376–1.40 m →
> it's a **tape entry error, not ours**. Harmless to the scale (tape mode uses the **median** of 15
> ratios → outlier ignored; tape scale 0.5595 vs survey 0.5537, ~1% apart, ratio CV 1.00%), but
> **EXCLUDE pair (85,113) in any per-pair tape analysis on field_A.** field_D not yet checked.
>
> **The survey code is NOT deleted** — `load_survey()` + the Umeyama-onto-survey path in
> `marker_scale.py`, and the survey-anchored `marker_gcp_ba.py` / `marker_gcp_lomo.py`, all stay on
> disk for later. They are simply **not part of the current route**. Re-enable if a session ever gets
> a trustworthy (total-station) survey.
>
> ### Two tape-only routes (both survey-free)
>
> **Route 1 — single COLMAP run (post-hoc, production default).**
> ```
> run_colmap.py (ONE normal SfM)  →  detect markers  →  triangulate (back-projection on poses)
>   →  tape scale  (k = median(tape_dist / recon_dist))  →  metric model
> ```
> COLMAP runs once; triangulation is just geometry; scale is a uniform multiply. No second
> optimization, so markers can't distort the reconstruction. Safe, simple, the workhorse.
>
> **Route 2 — second FULL COLMAP SfM with markers as tie-points (BUILT; experiment pending).**
> ```
> detect markers on input_uniform FIRST  →  feature_extractor → matcher → [inject markers into
>   database.db as tie-points: pixel + ID, NO coordinates] → mapper   (a full second SfM)
>   →  markers baked into SfM (arbitrary units)  →  tape scale on top
> ```
> Markers enter as **guaranteed-correct 2D correspondences** (same ID across images = a certain
> match), **not** as world coordinates — so this is survey-free. **Built (2026-06-26):**
> `src/preprocessing/inject_markers_to_db.py` writes marker keypoints + verified two-view matches into
> `distorted/database.db` (raw SQLite — pycolmap 4.0.4's Database is an abstract base that segfaults);
> `run_colmap.py` calls it between matcher and mapper when `inject_markers_json` is set (default "" =
> off, byte-identical normal run). **Validated** on field_A/20250609: injected 118 marker keypoints +
> 1279 match-edges → mapper registered **119/119** images (= baseline, no regression) and triangulated
> **114/118** marker keypoints into 3D points → markers are baked into the SfM. **Question still to
> measure:** does this improve **camera calibration** and/or **downstream 3DGS** versus Route 1?
> Likely upside is registration robustness on weak sessions (blurry / low-overlap). This is the
> survey-free version of the old "Arm C" (metric-from-scratch with survey — that variant stays
> deferred). **Run it:** detect on input_uniform → `run_colmap.py inject_markers_json=<that json>`.
> NOT yet wired into run_preprocessing (which runs colmap before detect); use the two scripts manually.
>
> Compare Route 2 vs Route 1 on: marker reproj (px), held-out marker error via tape pairs (mm),
> images registered (the robustness lever), and — once on Euler — 3DGS PSNR/SSIM/LPIPS on the same
> pinned split.

---

## STATUS & NEXT STEPS (as of 2026-06-27)

**ALIKED front-end + marker layer run across the whole 14-session phone series (2026-06-27):**
- **SfM front-end switched to ALIKED+LightGlue** (see [`PHONE_SFM_FRONTEND.md`](PHONE_SFM_FRONTEND.md)) — every session now registers 100 % into one connected model (SIFT fragmented). The marker layer below was triangulated against these new ALIKED reconstructions.
- **Marker layer run on all 14 sessions** (`run_preprocessing run_markers=true marker_scale_source=tape`, tape-only, quality guard active). All 14 → 6/6 markers triangulated. Outcome: **10 reliable tape-metric** (CV 0.7–2.7 %, ours-vs-tape 4.8–27 mm, `sparse_metric/` written), **3 flagged `reliable=False`** (CV ~7 %: both lisa-Pixel-6a sessions + field_D/20250706), **1 failsafe-blocked** (field_D/20250722 — only 2 quality markers; guard dropped 4 garbage ones with reproj up to 1440 px). **The (a) quality guard is proven in production: flagged 3 + blocked 1.**
- **✅ Marker-DETECTION gap SOLVED (field_D/20250722) — the brightness-based plate gate. → docs/MARKER_DETECTOR_LATE_SEASON.md.** Was: CCT v8 found markers in 0–11 views vs Agisoft's 9–22 on the same images. Scored our detector pin-by-pin against Agisoft's `marker_projections.csv` GT (63/63 cams map, GT in our pixel space; note all pins are `Pinned=True` which is *also* how Agisoft auto-detected coded targets export — pinned ≠ manual). **Gate attribution (92 GT projections): NCC proposes a candidate at 75/92 (82 %) — the matcher is fine — but the `white_surround` gate kills 57 of 75** (only 12 pass = our 13 % recall; killed markers score `white_surround_frac` ≈ 0–0.05). **Root cause:** the gate IDs the plate by *brightness* (bright-via-local-Otsu AND desaturated), and in late season the wheat is brighter than the grey/shaded plate → the real marker reads "not white" → rejected (and bright straw passes → FPs). **Fix:** a brightness-invariant **`plate_gate=lowsat`** (plate = achromatic / low HSV-saturation `S ≤ plate_s_max=110`, no brightness test; `detect_markers_v6.white_surround_frac` branches on `cfg.plate_gate`). Recall **13 % → 79 %**; the two FPs (77 @1103 px, 101 @737 px) **eliminated** (consensus outvotes them once it has the true cluster: 77 → 20 views/15 inliers @2.25 px, 101 → 18/11 @2.32 px); all 6 markers clean → **scale unblocked: 6/6, CV 2.24 %, 20.4 mm vs tape, `sparse_metric/` written.** 14-session batch (lowsat detect+tri vs white-gate baseline): **zero new FPs, 11 byte-equal, 2 improved (both 0722), 0 degraded** → lowsat is a strict superset; now the default (`white` kept for A/B). Decode cost negligible — the gate is only a pre-decode filter; real FP defense is decode + manifest + triangulation consensus + quality guard (all brightness-independent). See [[lowsat-plate-gate-idea]]. **RESIDUAL: NCC recall — oblique-template option BUILT (opt-in, default off, 2026-06-28).** 17/92 GT had no NCC candidate (steep late-season foreshortening the disk→ellipse the fronto-parallel template misses). Fix `oblique_templates=true` adds affine-squashed template copies (`_warp_oblique`+`build_template_bank`). Validated NCC recall 82%→89% but SITUATIONAL — extra views didn't become inliers once poses were good (markers already saturated); ~9× detection cost; rounder ratios (0.8) tested = +3 recall but ×2.5 candidate flood (not worth it). Default OFF, insurance for view-starved sessions. See MARKER_DETECTOR_LATE_SEASON.md §7.

**✅ POSE-ACCURACY vs Agisoft DONE 2026-06-28 → found open-walk DRIFT, fixed by matcher=exhaustive. → [`PHONE_SFM_POSE_ACCURACY.md`](PHONE_SFM_POSE_ACCURACY.md), [[project-phone-pose-accuracy]].**
- `compare_to_agisoft.py` showed sequential-matcher poses drift at the SWEEP ENDPOINTS (field_D/20250722 ends ~85–99 mm vs ~15 mm middle; the early-101 marker that wouldn't triangulate was the symptom — accurate 2D detection + 340 px reproj ⟹ bad POSE not bad detection). Mechanism: a single linear walk has no loop closure → drift accumulates down the chain (endpoints have MORE points yet worst poses → rules out feature starvation).
- **Fix: `matcher=exhaustive`** (all-pairs = free loop closure). 14-session rollout: median pose error vs Agisoft down 30–80 % (field_A/0618 51→10 mm), rotation ~halved. Now the phone default (O(N²); keep sequential for 170+ img sets).
- **Markers re-triangulated on the corrected poses → tape error dropped (field_A/0618 27→1.1 mm), 2 of 3 unreliable sessions flipped reliable → now 13/14 sessions reliable metric** (only field_D lisa still unreliable). The early-101 outliers became inliers for free.
- **Triangulation overlay made honest** (`triangulate_markers.py`): now draws ACCEPTED (green) vs RANSAC-REJECTED (red, "REJ") vs reprojected/snapped distinctly + bolder + legend, so rejected canopy detections no longer masquerade as good ones (they were all drawn green before).
- **Remaining (separate, non-drift):** the 2 lisa (Pixel-6a) sessions (camera/intrinsics issue) + field_D/20250618 (lone +11 % regression) — both ~110–600 mm regardless of matcher.

**Built + validated (2026-06-26):**
- **Tape-only metric scale** — `marker_scale.py` + `apply_metric_transform.py` have `scale_source: survey|tape`
  (survey default byte-identical; tape = scale-only similarity from tape distances, survey-free). Orchestrator
  threads `marker_scale_source` to steps 6+7 and auto-skips step 8 (GCP-BA is survey-anchored). Tested:
  field_A/20250609 tape scale 0.5595 vs survey 0.5537, ours-vs-tape 8.85 mm.
  **External corroboration** that this two-level design (post-hoc rescale vs prior-in-BA) is the right call —
  COLMAP issues #1051 (rescale, don't re-reconstruct = Level A) + #999 (no native distance-prior-in-BA; the
  mm-vs-px weighting is hard ⇒ coplanar markers barely move the bundle = our marginal Level B) + #2228/#1471/
  #2687: see [docs/MODEL_ALIGNER_SCALE_CHECK.md](MODEL_ALIGNER_SCALE_CHECK.md) §1.
- **Route 2 (markers baked INTO the SfM)** — `inject_markers_to_db.py` injects marker tie-points into
  `database.db`; `run_colmap.py` calls it (gated `inject_markers_json`); orchestrator `markers_in_sfm=true`
  runs a pre-COLMAP detect on `input_uniform` + injects. Validated: 20250609 → 119/119 + markers triangulated.
  Smooth one-command Route 2: `run_markers=true markers_in_sfm=true marker_scale_source=tape`.
- **Bugs fixed:** (1) orchestrator step-4 detect wrote `marker_detections_v8.json` but step-5 triangulate read
  `…_v8_manifest.json` (pre-existing name mismatch → triangulate crashed); (2) recap printed `scale_umeyama`,
  absent in tape mode (KeyError). Both fixed.

**OPEN — pick up here after compaction:**
- **(a) Marker-scale quality guard — ✅ BUILT + VALIDATED (2026-06-26).** Implemented in `marker_scale.py`
  as the single source of truth (`quality_thresholds` / `marker_quality_ok` / `filter_ours`), reused by
  `apply_metric_transform.py` (so the applied transform is anchored on the exact same markers the report
  trusted) AND by the orchestrator failsafe (`_count_quality_markers` now counts only quality-passing
  markers, not merely solved ones). Three per-marker gates (defaults, all configurable, set 0 to disable):
  `quality_min_parallax_deg: 10`, `quality_min_inlier_views: 4`, `quality_max_reproj_px: 8` — read from
  marker_points3d.json (parallax_deg / n_inliers / max_reproj_px). Plus tape mode gets a robust MAD
  outlier-reject on the per-pair ratios (`quality_ratio_mad_k: 3.5`) and a CV warning
  (`quality_warn_cv: 0.05` → `scale_reliable` flag). **Defaults are a NO-OP on every good run** (lowest seen
  parallax 36.7°, inliers 6, max-reproj 6.0px) and only cut the ~5°-parallax poisoners. `mad_k=0` keeps the
  scale byte-identical to before. The guard report (kept/dropped + thresholds) is embedded into
  marker_scale.json and metric_frame.json. **Validation:** on field_A/20250609 the guard kept all 6 markers
  and the MAD reject *automatically* dropped pair (85,113) — the documented tape entry error
  ([[project-tape-measure-error]]) — with zero hand-tuning → CV 0.60%, ours-vs-tape 6.20 mm; both
  marker_scale.py and apply_metric_transform.py produced the identical scale 0.558890 (single-source-of-truth
  confirmed). Config knobs added to `marker_scale.yaml`, `marker_metric.yaml`, `config.yaml`.
- **(b) The 76/113 registration gap — ✅ SOLVED (2026-06-26) by ALIKED+LightGlue front-end.** **HEADLINE
  RESULT:** swapping the feature front-end SIFT → **ALIKED + LightGlue** (learned detector + learned matcher,
  both NATIVE in this COLMAP 4.1 build) puts **all 113/113 images into ONE connected model** (vs SIFT's 4
  fragments / largest 76) and builds a reconstruction **denser than Agisoft**. Scored A/B/gold on
  field_A/20250603 with the new `src/analysis/analyze_sfm_connectivity.py`:
  | metric | SIFT (baseline) | ALIKED+LightGlue | Agisoft |
  |---|---|---|---|
  | sub-models | 4 | **1** | 1 |
  | largest model | 76/113 | **113/113** | 113/113 |
  | 3D points | 4,472 | **62,351** | 49,592 |
  | obs/image | 202 | **1,474** | 900 |
  | strong pairs (≥30) | 285 | 773 | — |
  | reproj err | 1.31px | 1.39px | — |
  The diagnosis (below) predicted exactly this: the bottleneck was sparse features → sparse triangulation →
  marginal frames can't register. ALIKED's denser, better-localized learned features + LightGlue's
  ambiguity-resolving matcher fix it. **Cost:** GPU extraction 44s + LightGlue *exhaustive* matching ~12 min +
  mapper 110s; the 12-min match is the long pole → try **sequential matching** for production (suits the
  continuous sweep). **Run it:** `--FeatureExtraction.type ALIKED_N16ROT --FeatureMatching.type
  ALIKED_LIGHTGLUE` (harness `scratchpad/aliked_run.sh`). **LOCAL ENV GOTCHA:** COLMAP's ONNX provider was
  built for CUDA 12 but the box has CUDA 13 → ALIKED GPU aborts (`libcublasLt.so.12` missing). FIX (no env/torch
  change): `pip install --target <scratch> nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
  nvidia-cufft-cu12` then prepend those `nvidia/*/lib` dirs to `LD_LIBRARY_PATH` for the COLMAP call. **SIFT
  kept untouched as the comparison baseline.** TODO: wire ALIKED as a `camera`/front-end option in
  `run_colmap.py`; run on the other 13 sessions; check pose accuracy vs Agisoft (compare_to_agisoft). **The
  fragmentation diagnosis that led here ↓ (kept for the record):**
- **(b-diagnosis) The 76/113 gap was reconstruction FRAGMENTATION, not registration failure.** On 20250603 our COLMAP actually
  registered **110/113** images, but the mapper split them into **4 disconnected sub-models** (76 + 12 + 11 +
  11); `run_colmap.py` keeps only the largest (76). The fragments are **temporally interleaved** with model 0
  (all inside one continuous 150300→150432 sweep, e.g. sub-model 3 = 150341–150349 sits *inside* model 0's
  span) → same physical area, but the incremental mapper repeatedly lost+re-acquired the thread on repetitive
  wheat and spawned a new sub-model each time. A clean no-marker baseline fragments identically (76 + 30 + 11),
  so it's NOT caused by marker injection. **Neither rescue tool works:** `model_merger` fails (the sub-models
  share ZERO co-registered images, so it has nothing to align on); `image_registrator` adds 0 (the missing
  frames see only 9–21 of model 0's 3D points, far below the ~30 needed). **Root cause (from the database):**
  between an 11-frame fragment and the 76-frame model there are only **10 geometrically-verified pairs** (of
  836 possible), and just **4** clear the ~30-inlier bar — a hair-thin bridge; within model 0 connections are
  dense. So the bottleneck is **sparse cross-cluster feature matching on repetitive wheat** = the real
  phone-SfM-vs-Agisoft front-end question (Agisoft's better features/markers-as-tie-points keep it one model).
  **NEXT (deferred at user request — CPU-heavy): test a stronger front-end.** Started COLMAP's repetitive-scene
  recipe (CPU SIFT `estimate_affine_shape=1` + `domain_size_pooling=1` + `max_num_features=16384` + exhaustive
  `guided_matching=1`); the 113-image extraction finished (~5 min) but CPU guided matching was ~30–60 min on
  all 8 cores → stopped. The extraction DB is cached at
  `scratchpad/regexp/strong/database.db` → **resume by running only GPU `exhaustive_matcher`
  (`--FeatureMatching.guided_matching 1`) + mapper on it (~1–2 min)**. Experiment harness:
  `scratchpad/reg_experiment.sh`. Other levers if SIFT is insufficient: **ALIKED** (learned features, native in
  this COLMAP 4.1 build — no external hloc) or hloc/SuperPoint+LightGlue. See
  [`COMPARE_TO_AGISOFT_RESULTS.md`](COMPARE_TO_AGISOFT_RESULTS.md).
- **Route 1 vs Route 2 experiment — capability built, NOT yet MEASURED.** A/B on the same sessions (toggle
  `markers_in_sfm`): compare registration count, marker reproj, and downstream 3DGS PSNR/SSIM/LPIPS. The weak
  sessions are where Route 2 *might* help.
- **3DGS bridge — NOT built.** `run_reconstruction.py` cannot yet train on `sparse_metric/`. Add a
  `use_metric_sfm` flag (mirror `use_agisoft_sfm`) to train 3DGS on the tape-metric model, then train on Euler
  (3DGS = Euler only). This is the path to actual metric phenotyping (measure a marker distance in the trained
  model, confirm it matches tape).

---

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

**Which metric reference you need — three tiers (what unlocks "real metres").** COLMAP alone produces only *shape* (arbitrary units). You need SOME real-world length reference to recover *size*; how much you measure decides how much you get:

| reference you have | what it unlocks | needed for steps |
|---|---|---|
| **Survey coordinates** (`field_<L>_coordinates.txt`) | full metric scale **+ absolute georeferencing** (exact CH1903 position) + orientation | metric scale (6), Flavour 1 (7), Flavour 2 (8), LOMO experiment |
| **Tape distances only** (hand-measured marker-to-marker) | **scale/size only** (no absolute world position) — a cheaper manual fallback | could drive scale (6) without survey |
| **Neither — but markers have a KNOWN printed size** (15 cm square / 13 cm circle, spec PDF) | **scale/size only**, with *zero* field measurement — a "free" ruler baked into the marker | not yet implemented (forward-looking) |
| **No real-world reference at all** | nothing — stays arbitrary units; markers add nothing over COLMAP's natural feature points | — |

So detect (4) + triangulate (5) need **no** measurements (markers come out in arbitrary units); only the metric steps (6–8) + the LOMO experiment need survey. Without survey but with tape → scale only. Without either, the markers' own known physical size *could* still give scale (not built yet).

## 1b. Reference availability — cases, survey-quality detection, and the soft-GCP question

> **Working assumption:** for the current demoanlage data we HAVE both survey + tape on every field. This
> section records what happens when we don't, because future captures may lack one — and because the
> answers changed our priorities. (Raised + discussed 2026-06-23.)

**A. What actually runs in each case** (current code behaviour; marker steps are fail-soft, so a missing
file warns + skips rather than crashing):

| you have | runs | you get | georeferenced? |
|---|---|---|---|
| **survey + tape** (current) | all steps 4–8 + tape cross-check + LOMO | full metric model, validated two ways | ✅ yes |
| **survey, no tape** | all steps 4–8 (tape compare skipped — coded optional) | full metric model; lose the independent tape check | ✅ yes |
| **tape, no survey** | detect + triangulate run; metric steps **skip** TODAY | markers in arbitrary units — *but Tier-2 scale is possible*, see C | ❌ size-only even when built |
| **neither** | detect + triangulate only | markers in arbitrary units, no metric scale | ❌ |

**The metric MODEL needs the *survey* (absolute XYZ)** — `apply_metric_transform.py` + `marker_gcp_ba.py`
use it. **The tape is never required to build the model** — it is a validation cross-check only
(`marker_scale.py` tolerates a missing tape xlsx).

**B. When do we know a field's survey is GOOD vs BAD?**
- **Before re-running COLMAP-with-markers — IF we have the tape.** The Step-3 check
  (`marker_scale.py`) runs on the *already-finished* reconstruction (post-hoc triangulation, no re-run)
  and compares survey vs the **independent tape**. Disagreement = suspect survey. This is how we caught
  field_A (tape↔survey 17 mm) vs field_D (8 mm) *without* any marker-anchored BA.
- **Only after — if survey is our ONLY reference.** Then the **LOMO test** (which IS the marker-anchored
  BA) is what exposes it: hold one marker out, see if anchoring the rest predicts it well. Verdict =
  output of the run.
- So: **tape (or Agisoft) ⇒ known in advance; survey-only ⇒ found out by running it.**

**C. Tape-only metric scale (Tier 2) — possible, NOT yet wired.** With tape but no survey we *can* still
get metres: marker distances in COLMAP units ÷ same distances from the tape = scale factor → multiply the
model. Gives **size only**, not georeferencing — **but phenotyping (real-mm head sizes) only needs size**,
so tape-alone would suffice for the thesis end-goal. `marker_scale.py` already computes a distance-ratio
scale; what's missing is a Flavour-1 variant that applies a *pure scaling* (no absolute survey anchor).
**TODO if a field ever lacks survey.**

**D. The soft-GCP question (demoted by the tape gate).** The LOMO experiment showed hard-anchoring helps
on good-survey fields and hurts on bad ones (§ see `MARKER_COLMAP_RERUN_EXPERIMENT.md`). Two ways to act:
- **Tape gate (cheap, enough for now) — ✅ IMPLEMENTED** (`run_preprocessing.py`, `tape_gate=true`; see
  §8): if tape agrees with survey (≤ 12 mm) → run the anchored BA (Flavour 2); if it disagrees → stay
  post-hoc (Flavour 1). All-or-nothing per field. Works because our fields are cleanly good (field_D) or
  bad (field_A). Writes `logs/metric_choice.json`.
- **Soft/weighted GCPs (general, more work):** per-marker uncertainty so good markers pull and bad ones
  are down-weighted. Needed only for (i) *mixed-quality within one field* (5 good + 1 bad — the gate
  would discard the whole field) or (ii) *no-tape* sessions (no cheap pre-check). What Agisoft does
  natively (marker "accuracy").
- **Decision:** the tape gate handles our current data, so **soft GCPs drop from "next" to "later, if
  needed"** — revisit when we hit a mixed-quality or tape-less field.

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

`marker_projections.csv` columns: `Marker, Camera, X, Y, Pinned` — i.e. "marker *M* is at pixel (X,Y) in photo *Camera*"; `Pinned=True` = genuinely detected by Agisoft, `False` = back-projected. **Coordinates are w.r.t. the UNDISTORTED images** (use `cv2.undistortPoints` for the distorted variants). **NOTE (checked all CSVs): the export is 100 % `Pinned=True`** — 4228/4228 phone rows + all FIP rows, zero `False`. Agisoft only exports the **measured** pins and omits back-projected ones, so the column is a **no-op filter in our data** (we read it in `overlay_agisoft_markers.py` but it never discriminates). It is NOT a "manual vs auto" flag — auto-detected coded targets export `True` too. Our own pipeline expresses the same found-vs-predicted split not as `Pinned` but as the observation `src` field in `triangulate_markers.py`: `detected`+inlier / `snapped` (measured, code-corrected by location) = "found"; `reprojected` = predicted.

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
**Measured timing (FULL pipeline, field_A/20250609):** uniform 0s / COLMAP 1:55 / compare 1s / detect 2:08 /
triangulate 0:16 / scale+Flavour 1+Flavour 2 ≈ 1s → total 4:24. **COLMAP SfM (~44%) + CCT detection (~48%) =
~92%**; everything else negligible. Flavour 2 (GCP-BA) is **1 second** — it does NOT re-run COLMAP, just one
pycolmap BA pass on the finished model. **CCT detection then parallelised → 2:08 → 0:22 (5.8×)** (see
`MARKER_DETECTION_CCT.md`; `num_workers`, default 8).

**TAPE GATE (`run_preprocessing.py`, default OFF — `tape_gate=true` to enable).** Auto-picks the metric model
from survey trustworthiness, operationalising the Arm-A finding (anchoring HELPS on good survey, HURTS on bad).
After Step 3 it reads `tape_vs_survey_mean_abs_mm` from `marker_scale.json`: **≤ `tape_gate_threshold_mm`
(default 12) → survey GOOD → run Flavour 2 GCP-BA (step 8), chosen model = `sparse_metric_gcp/`; > threshold →
survey SUSPECT → skip step 8, keep Flavour 1 (`sparse_metric/`)**. Writes `logs/metric_choice.json`
(`chosen_model`, `survey_quality`, `tape_vs_survey_mm`, `ran_gcp_ba`) for a downstream consumer (e.g. which
model to feed 3DGS) + shows the choice in the recap. Verified: field_D 8.2 mm → GOOD/`sparse_metric_gcp`,
field_A 17.1 mm → SUSPECT/`sparse_metric`. Graceful: no tape data → warns + doesn't block (runs GCP-BA). This
is the cheap binary alternative to soft GCPs (§1b D).

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
