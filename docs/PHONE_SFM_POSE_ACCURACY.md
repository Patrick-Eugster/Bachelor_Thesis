# Phone SfM pose accuracy: open-walk drift, and the exhaustive-matching fix

**TL;DR** — After ALIKED solved *registration* (one connected model, 100% of images — see
[PHONE_SFM_FRONTEND.md](PHONE_SFM_FRONTEND.md)), a second-order problem remained: **camera-pose
accuracy**. Our `sparse/0` cameras drifted vs Agisoft, worst at the **ends of each sweep** (field_D/
20250722 endpoints ~85–99 mm off, middle ~15 mm). Cause: a single linear walk has **no loop closure**,
so `matcher=sequential` (each frame matched only to its ±25 time-neighbours) lets error accumulate down
the chain. Fix: **`matcher=exhaustive`** (all-pairs) supplies the missing long-range cross-links and
closes the open walk. Across **14 sessions** it cut median pose error vs Agisoft (e.g. field_A/20250618
**51 → 10 mm**, field_A/20250603 **111 → 49 mm**), roughly halved rotation error, and — re-triangulating
the markers on the corrected poses — tightened the metric scale (field_A/20250618 tape error **27 → 1.1
mm**), taking **11/14 → 13/14 sessions to reliable metric**. Exhaustive is now the phone default.

This was found while chasing a marker issue; the diagnostic chain is the useful part.

---

## 1. How it surfaced — a marker that wouldn't triangulate

On `field_D/20250722`, after the [lowsat recall fix](MARKER_DETECTOR_LATE_SEASON.md), marker **101**
was still detected in the first ~6 frames of the sweep but **rejected by triangulation** (≈340 px
reprojection). The detections were **0 px from Agisoft's GT pins** — i.e. *correct*. An accurate 2D
detection that reprojects 340 px wrong can only mean one thing: **the camera pose for those frames is
wrong**. The marker triangulation was acting as an unintended **pose-consistency probe**.

Key distinction it forced: a reprojection error can come from a bad *detection* **or** a bad *pose*.
We separated them by comparing the detection to GT (0 px → detection fine) → the residual is **pose**.

## 2. Measuring it — `compare_to_agisoft.py`

`compare_to_agisoft.py` Umeyama-aligns our `sparse/0` onto Agisoft's metric `agisoft/sparse/0` and
reports, per camera, **translation error (mm)** and **rotation error (deg)**. On `field_D/20250722`
(sequential): median 35 mm / 1.7°. Per-camera, sorted by frame:

| segment | median translation err |
|---|---|
| first 10 frames (sweep **start**) | **85.5 mm** |
| middle | 15–30 mm |
| last 10 frames (sweep **end**) | **81.8 mm** |

The exact frames where marker 101 was rejected (182044–048) ranked **65th–75th of 76** — the worst
cameras in the model. Inference confirmed: the **endpoints are drifted**.

## 3. The mechanism — open-walk drift, not feature starvation

The obvious guess (endpoints are starved of features → weak pose) is **wrong**. Observations-per-frame:

| | 3D observations / frame | pose error |
|---|---|---|
| first 10 (start) | **1390** | 85 mm (worst) |
| middle | 1104 | **15 mm (best)** |
| last 10 (end) | 1227 | 92 mm (worst) |

The endpoints have **more** points than the middle, yet the worst poses. That inverse pattern is the
signature of **drift along an open trajectory**: each pose is solved relative to its neighbours; you
walked the row **once in a line**, so the two ends never see each other → no constraint ties them
together → small per-frame errors accumulate down the chain like a tape measure laid end-to-end. The
middle sits near the global-alignment anchor (fits well); the ends are farthest down the chain (drift
piles up) — regardless of local point count. A **loop** (revisiting a spot) would cancel it; a single
pass has none.

## 4. The fix — exhaustive matching (= free loop closure)

`sequential` matches each frame only to its ±`sequential_overlap` time-neighbours. **`exhaustive`
matches every pair**, so any non-adjacent frames that share ground get linked — exactly the long-range
cross-links an open walk lacks. (Loop *detection* is a cheaper approximation that only fires on an
actual revisit; exhaustive is the brute-force superset and needs no revisit.)

On `field_D/20250722`, sequential → exhaustive:

| segment | sequential | **exhaustive** |
|---|---|---|
| first 10 (start) | 85.5 mm | **28.5 mm** (−67 %) |
| last 10 (end) | 81.8 mm | **14.0 mm** (−83 %) |
| all (median) | 35.1 mm | **23.9 mm** |
| focal vs Agisoft | +4.1 % | **+2.2 %** |

And the early-101 detections that were rejected at ~340 px became **all inliers at 0.8–3.7 px** — the
markers cleaned up *for free* once the poses were right (they were never false positives, just
correct detections in drifted frames).

## 5. 14-session rollout — exhaustive wins broadly

Re-ran every session exhaustive and re-compared to Agisoft (median translation error, mm):

| session | imgs | seq | exh | Δ | exh time |
|---|--:|--:|--:|--:|--:|
| field_A/20250618 | 93 | 51.0 | **9.7** | −81 % | 9m47 |
| field_D/20250627 | 88 | 22.7 | **10.8** | −52 % | 8m47 |
| field_A/20250715 | 96 | 16.2 | **9.2** | −43 % | 10m23 |
| field_A/20250722 | 64 | 25.4 | **16.1** | −37 % | 5m07 |
| field_A/20250627 | 84 | 28.1 | **17.0** | −39 % | 8m01 |
| field_D/20250603 | 100 | 45.2 | **22.1** | −51 % | 11m11 |
| field_D/20250722 | 76 | 35.3 | **24.1** | −32 % | 6m50 |
| field_D/20250715 | 96 | 31.8 | **25.0** | −21 % | 10m32 |
| field_A/20250706 | 93 | 30.8 | **25.6** | −17 % | 9m19 |
| field_A/20250603 | 113 | 111.4 | **48.8** | −56 % | 14m04 |
| field_D/20250706 | 75 | 120.4 | **79.5** | −34 % | 6m26 |
| field_D/20250618 | 127 | 107.8 | 119.2 | **+11 %** | 17m43 |
| field_A/20250613 lisa | 183 | 188.9 | 149.6 | −21 % | 34m22 |
| field_D/20250613 lisa | 172 | 735.0 | 603.1 | −18 % | 30m16 |

**13 of 14 improved**, rotation error roughly halved on most. Groups:
- **Clean wins → ~10–25 mm, <1° (production-ready):** 0618, 0715, both 0627, both 0722, field_D/0603.
- **Big improvement, still moderate:** field_A/0603 (111→49), field_D/0706 (120→80) — drift was much
  of it, something else remains.
- **Not a drift problem:** the **two lisa (Pixel-6a) sessions** (150 mm, 603 mm/30°) barely move — a
  separate-camera/intrinsics issue, *not* matching. **field_D/20250618** is the lone regression
  (+11 %, ~110→119 mm) — its own problem; both seq and exh are bad there.

## 6. Marker scale improved too (the downstream payoff)

Re-triangulating the markers on the exhaustive poses (lowsat detections reused) and re-scaling
(tape-only). Marker tape error and CV, sequential → exhaustive poses:

| session | tape err | CV | reliable |
|---|---|---|---|
| field_A/20250618 | 26.7 → **1.1 mm** | 2.6 → 0.1 % | ✓ |
| field_A/20250603 | 27.1 → **8.9 mm** | 2.6 → 0.8 % | ✓ |
| field_A/20250627 | 21.5 → **4.3 mm** | 1.8 → 0.5 % | ✓ |
| field_D/20250603 | 17.5 → **6.2 mm** | 1.8 → 0.7 % | ✓ |
| field_D/20250618 | 8.8 → **2.3 mm** | 0.9 → 0.3 % | ✓ |
| field_A/20250613 lisa | 72 → **21 mm** | 7.0 → 2.0 % | **False → True** |
| field_D/20250706 | 64 → **52 mm** | 7.4 → 5.0 % | **False → True** |
| field_D/20250613 lisa | 65 → 56 mm | 7.0 → 5.5 % | False (still) |

**11/14 → 13/14 sessions now have reliable metric scale.** Note field_A lisa became reliable
(CV 2.0 %, 21 mm) even though its *pose-vs-Agisoft* stayed ~150 mm — markers measure **internal**
consistency (CV of tape ratios), which exhaustive fixed, while the global Umeyama comparison is thrown
off by the different camera. So the lisa "problem" is narrower than the pose number implied. Only
**field_D lisa** remains unreliable (CV 5.5 %, just over the 5 % bar).

## 7. Decision & guidance

- **`matcher=exhaustive` is now the phone default** (`configs/preprocessing/colmap.yaml`). Same
  registration as sequential, materially better poses + metric scale.
- **Cost is O(N²):** ~5 min @ 64 img, ~14 min @ 113 img, ~34 min @ 183 img. For **large sets (170+)**
  prefer `matcher=sequential` with a higher `sequential_overlap` (or loop detection) to dodge the N²
  blow-up — the only large sessions here are the broken lisa ones anyway.
- **Capture-time cure (future shoots):** walk the row out *and back*, or weave slightly — any overlap
  between non-adjacent frames gives loop closures for free and prevents endpoint drift at the source.

## 8. Open / separate issues (not drift)

- **Both lisa (Pixel-6a) sessions** — different camera; pose-vs-Agisoft stays high. field_A lisa is
  nonetheless usable for *scale* now; field_D lisa is not (CV 5.5 %). Needs its own look (intrinsics /
  reference frame), not matching.
- **field_D/20250618** — the one exhaustive regression (~110→119 mm); both matchers land ~110 mm, so a
  distinct problem.
- **NCC recall residual** (separate, from the marker work) — steep oblique late-season plates the
  fronto-parallel templates miss; would tighten markers further but isn't pose-related.

## 9. Files / how to reproduce

```bash
# pose error vs Agisoft (per-camera translation mm + rotation deg)
python src/preprocessing/compare_to_agisoft.py field=field_D plot=20250722

# exhaustive re-run (ALIKED); CUDA-12 onnx lib dir for ALIKED on a CUDA-13 box (see PHONE_SFM_FRONTEND.md)
python src/preprocessing/run_colmap.py field=field_D plot=20250722 \
  front_end=aliked matcher=exhaustive aliked_cuda12_libdir=<dir>

# re-triangulate + re-scale markers on the new poses (reuse existing detections)
python src/preprocessing/run_preprocessing.py field=field_D plot=20250722 \
  run_uniform=false run_colmap=false run_markers=true run_marker_detect=false marker_scale_source=tape
```

Related: [PHONE_SFM_FRONTEND.md](PHONE_SFM_FRONTEND.md) (registration/ALIKED),
[MARKER_DETECTOR_LATE_SEASON.md](MARKER_DETECTOR_LATE_SEASON.md) (lowsat recall fix),
[MARKER_INTEGRATION_PLAN.md](MARKER_INTEGRATION_PLAN.md) (marker layer).
