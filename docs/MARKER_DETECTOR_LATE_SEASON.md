# Marker detector: the late-season recall fix (brightness-invariant plate gate)

**TL;DR** — Our CCT marker detector (v8) collapsed on late-season sessions: on `field_D/20250722` it
localized only **13%** of the markers Agisoft pins, leaving the metric-scale step **blocked** (2 of 6
markers were canopy false positives at 600–1100 px reprojection). The cause was a single gate that
identifies the marker plate by **brightness** ("the plate is the bright thing"). In late season the
wheat is *brighter* than the grey/shaded plate, so the gate threw the real markers away. Replacing it
with a **brightness-invariant, saturation-based** gate (`plate_gate=lowsat`) lifts GT recall **13% →
79%**, eliminates the false positives, and unblocks the session — with **zero new false positives
across all 14 sessions**. lowsat is now the default; `white` stays for A/B.

---

## 1. Symptom

`field_D/20250722` (late season, dense golden canopy): Agisoft pins all 6 coded targets in 9–22 views
each; our v8 detector ended with only 2–11 manifest views per marker, and after triangulation **two
markers (77, 101) were false positives** — canopy patches that decoded to those codes at wrong
locations and triangulated to **609–1103 px reprojection** garbage. The quality guard correctly
rejected them, which left **< 4 good markers → metric scale blocked**.

## 2. Where it missed — gate attribution against GT

Agisoft ships `marker_projections.csv` (per-image marker pixel positions) for the 0722 sessions. All
63 GT cameras map directly onto our images, and the GT X,Y are in our image pixel space — so we can
score our detector pin-by-pin. The recall path is:

```
NCC template match  →  contrast guard  →  white/plate surround gate  →  CCT decode  →  manifest filter
```

Attribution across the 92 GT projections on `field_D/20250722`:

| stage | count | reading |
|---|---:|---|
| NCC proposes a candidate at the marker | **75 / 92 (82 %)** | the template matcher *does* find the plates |
| …then killed by the **`white_surround` gate** | **57** | ← the dominant failure |
| …killed by the contrast guard | 6 | |
| …pass all gates | **12 (13 %)** | = our final recall |

The killed markers scored `white_surround_frac` ≈ **0.00–0.05** — essentially *zero* "white" pixels
around a plainly-visible plate. **NCC recall is fine; the white-plate gate is the bug.**

## 3. Root cause — identifying the plate by brightness

`white_surround_frac` ([src/preprocessing/detect_markers_v6.py](../src/preprocessing/detect_markers_v6.py))
defined the plate as **"BRIGHT (local-Otsu split) AND desaturated (S ≤ 70)"**. Both halves break in
late season:

1. **Brightness:** "bright" is defined *relative to the local window* via Otsu. When bright sunlit
   straw is in the window, Otsu locks its threshold onto the straw, so the **darker grey plate falls
   below it → counted as not-white.**
2. **Saturation:** a shaded plate picks up a faint bluish cast, pushing its saturation **above the
   70 ceiling** → fails the second half too.

So a genuinely visible marker (even a clean, near-frontal one) is rejected, while bright bleached
straw passes — producing **both** the misses *and* the canopy false positives, from one wrong
assumption: *the plate is the bright thing.* In late season it is the **darker** thing.

## 4. The idea — "lowsat" (brightness-invariant plate gate)

Stop keying on brightness; key on **colorlessness**, which doesn't change with light.

In **HSV**, **saturation** measures how pure/vivid a color is, *independent of brightness*: a dark-grey
and a bright-white pixel both have near-zero saturation (grey/white are "colorless"); a golden straw is
strongly saturated whether sunlit or shaded. The marker plate is **achromatic** (neutral grey/white →
low S at any light level); the wheat is **golden** (high S, always). So:

> **plate = the low-saturation (achromatic) region of the surround — no brightness test at all.**

This is brightness-invariant by construction: bright-white plate and dark-grey plate both pass; golden
straw fails regardless of how bright the sun makes it.

`white_surround_frac` now branches on `cfg.plate_gate`:
- **`white`** (legacy): `bright(local-Otsu) AND S ≤ white_s_max` — unchanged, byte-identical to before.
- **`lowsat`** (default): `S ≤ plate_s_max` only.

**The saturation ceiling matters:** at `S ≤ 70` (the old white ceiling) lowsat still only recovers
14 % — shaded blue-grey plates exceed it. At **`S ≤ 110`** it passes **73 / 92 (79 %)** — i.e. ~every
marker NCC found. `plate_s_max: 110` is the default.

```yaml
# configs/preprocessing/detect_markers_v8_cct.yaml
plate_gate: lowsat     # "white" = legacy bright+desaturated; "lowsat" = achromatic only (default)
plate_s_max: 110       # HSV S ceiling for lowsat (70 too tight for shaded plates; 110 validated)
```

## 5. Why a looser gate adds NO false positives

The plate gate is only a **pre-decode filter** — it decides *what to attempt decoding*, not what
becomes a final marker. The real false-positive defense is **downstream and brightness-independent**:

1. **CCT decode** — a candidate must decode to a valid 12-bit code;
2. **manifest filter** — that code must be one of the 6 deployed targets `{77,85,89,101,105,113}`;
3. **multi-view triangulation consensus (RANSAC)** — the detection must agree in 3D across frames;
4. **the quality guard** — reproj / parallax / inlier-views.

A canopy patch can decode to a manifest code in *one* frame, but it cannot be 3D-consistent with the
true marker across *many* frames. Crucially, **consensus needs a critical mass of TRUE detections to
outvote the canopy noise — and the white gate was discarding exactly those.** With lowsat feeding
consensus the real detections, the former FPs become rejected outliers:

`field_D/20250722` codes 77 and 101, white-gate → lowsat:

| code | white-gate | lowsat |
|---|---|---|
| 77 | 2 views, **1103 px** ❌ FP | 20 views, 15 inliers, **2.25 px** ✅ |
| 101 | 3 views, **737 px** ❌ FP | 18 views, 11 inliers, **2.32 px** ✅ |

The outlier (canopy) detections are still produced, but RANSAC now rejects them (77: 15 of 20 kept;
101: 11 of 18). Result: all 6 markers clean (median reproj 1.1–2.7 px), **scale unblocked** —
6/6 markers, CV 2.24 %, 20.4 mm vs tape, `sparse_metric/` written.

## 6. 14-session validation — strict superset, zero regressions

Re-ran detect(lowsat) + triangulate on every session, diffed against the white-gate baseline
(quality marker = inliers ≥ 4 **and** median reproj ≤ 8 px):

| outcome | sessions |
|---|---|
| **unchanged** (genuinely-bright; gates agree, ≤ 1 inlier drift) | 11 sessions |
| **improved** | `field_A/20250722` 5/6 → **6/6** (+32 inliers) · `field_D/20250722` 2/6 → **6/6** (+50 inliers, 2 FPs removed) |
| **degraded / new false positive** | **none** |

So lowsat is byte-equivalent where "white" already worked and rescues the late-season sessions where
it didn't. Decode cost is negligible (31 s vs 28 s on 0722 despite more candidates — the manifest +
concentric-decode reject the extras cheaply).

## 7. Residual limit (separate follow-up)

The remaining gap is **NCC recall**, not the gate: 17 of 92 GT markers on 0722 had *no* NCC candidate
at all (e.g. code 89 gets only 4 views), from steep oblique late-season viewing angles foreshortening
the circular ring into a thin ellipse the fronto-parallel templates miss. Fixing that means a stronger
*proposal* stage (oblique/affine or multi-orientation templates) — a smaller, independent task. The
gate fix already takes 0722 from blocked to a reliable 6/6 metric scale.

## 8. Files

- Gate: [src/preprocessing/detect_markers_v6.py](../src/preprocessing/detect_markers_v6.py) `white_surround_frac` (mode branch) + `detect_one`.
- Config: [configs/preprocessing/detect_markers_v8_cct.yaml](../configs/preprocessing/detect_markers_v8_cct.yaml) `plate_gate`, `plate_s_max`.
- GT scorer: [src/preprocessing/score_markers_vs_gt.py](../src/preprocessing/score_markers_vs_gt.py) (RED = GT missed, MAGENTA = stray candidate).
- Context: [docs/MARKER_INTEGRATION_PLAN.md](MARKER_INTEGRATION_PLAN.md) item (b) / late-season gap.
