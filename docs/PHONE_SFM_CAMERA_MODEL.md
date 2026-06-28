# Phone SfM camera model — is `SIMPLE_PINHOLE` still the right default under ALIKED?

**TL;DR** — The old reason for defaulting to `SIMPLE_PINHOLE` ("`PINHOLE`/`OPENCV` collapse to 2/93 on
repetitive wheat") was a **SIFT-era** result and is now **obsolete**: under the ALIKED+LightGlue
front-end, `SIMPLE_PINHOLE`, `SIMPLE_RADIAL` and `OPENCV` **all register 93/93** on `field_A/20250618`.
The richer models no longer fail — they just don't help: pose-vs-Agisoft is noise-level (~1 mm / ~0.3°),
Chamfer identical (~22 mm), while they cost 7–17 % more time, crop the undistorted image, and drift the
focal +2.8 %/+3.7 % vs Agisoft. **Honest caveat:** that focal agreement is partly circular (Agisoft also
uses `SIMPLE_PINHOLE`); the Agisoft-*independent* metric (internal reprojection error) slightly favours
the richer models, so `SIMPLE_PINHOLE` is the robust **default**, not the provably *most accurate* model.
The deciding test (does removing corner distortion improve 3DGS PSNR?) is **unrun**.

---

## 1. Why re-test

ALIKED+LightGlue landed only in the latest commit; the camera-model decision predates it. The original
justification was: weak/repetitive wheat features can't constrain extra distortion params, so the solver
diverges — empirically `SIMPLE_PINHOLE` registered 63/93 while `PINHOLE`/`OPENCV` got **2/93**. But ALIKED
attacks exactly that weakness (far more, far cleaner matches), so the collapse may no longer happen.

## 2. Setup

One clean plot (`field_A/20250618`, 93 images), front-end **ALIKED**, **exhaustive** matching,
`single_camera=true`. Three COLMAP runs — `camera = SIMPLE_PINHOLE | SIMPLE_RADIAL | OPENCV` — each into
its own scratch dir (real `sparse/0` untouched), then `compare_to_agisoft.py` per run. Wall-clock timed.

## 3. Results

| camera | reg/93 | colmap_s | trans median | rot median | Chamfer | focal vs Agisoft | cx/cy offset | undist. size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **SIMPLE_PINHOLE** | **93/93** | 593 | 9.51 mm | 0.561° | 21.9 mm | **−0.08 %** | 0.0 / 0.0 | 4032×3024 |
| **SIMPLE_RADIAL** | **93/93** | 637 | 9.37 mm | 0.497° | 22.7 mm | +2.78 % | 0.0 / 0.0 | 3942×2956 |
| **OPENCV** | **93/93** | 691 | 8.41 mm | 0.854° | 22.6 mm | +3.70 % | 0.0 / 0.0 | 3881×2923 |

Internal reprojection error (Agisoft-independent — see §5) and the distortion COLMAP actually fit:

| camera | internal reproj median | fitted distortion |
|---|---:|---|
| SIMPLE_PINHOLE | 1.323 px | — |
| SIMPLE_RADIAL | 1.299 px (−1.8 %) | k1 = +0.035 |
| OPENCV | **1.263 px (−4.5 %)** | k1=+0.050, k2=−0.037, p1=+0.003, p2=−0.008 |

## 4. Reading the numbers

1. **The collapse is gone.** All three register 93/93 under ALIKED. The SIFT-era "2/93" justification is
   obsolete — ALIKED's matches constrain the distortion params fine.
2. **Pose gain is noise-level.** trans/rot vs Agisoft differ by ~1 mm / ~0.3° (`SIMPLE_RADIAL` marginally
   best overall, `OPENCV` best translation but worst rotation); Chamfer is identical (~22 mm).
3. **The richer models cost more for less convenience:** 7–17 % slower, and undistortion *crops* the image
   (4032 → 3942 → 3881 px) because it warps away distortion and trims the invalid border.
4. **Principal point stays centered (0.0/0.0) for ALL models** — COLMAP doesn't refine the principal point
   by default (`Mapper.ba_refine_principal_point=0`). So switching camera model alone never reintroduces
   the [pixel-shift](PIXEL_SHIFT_BUG.md) sensitivity on phone (that needs an *off-center* principal point,
   which only appears if you also enable principal-point refinement). Phone stays immune regardless.

## 5. The circularity caveat (why "matches Agisoft" isn't proof)

The focal comparison is **referenced to Agisoft, which itself used `SIMPLE_PINHOLE`** — so both ignore
lens distortion identically and trivially agree. "Focal −0.08 % vs Agisoft" proves *consistency with our
benchmark*, not *physical correctness*. To judge correctness we need a metric that doesn't reference
Agisoft: the **internal reprojection error**.

**Internal reprojection error** = take the model's *own* solved 3D points, project each back into the
images using the model's *own* camera + poses, and measure the pixel gap to where the feature was actually
observed. It's the model grading *itself* on its own data — no external reference. A camera model that
matches the true lens explains the observations better, so it lands a lower reprojection error. Here the
richer models *do* score lower (1.323 → 1.299 → 1.263 px), which means **there is real lens distortion**
(k1≈0.05 ⇒ ~50–60 px radial shift at the extreme image corner, only a few px over most of the frame) and
`OPENCV` is marginally *more* correct than `SIMPLE_PINHOLE`. (Caveat on the caveat: more parameters can
always lower reprojection error by a hair via overfitting — but a clean physical interpretation here, plus
the modest size, makes "small real distortion" the honest reading.)

## 6. Verdict + open question

`SIMPLE_PINHOLE` stays the **default** — but for an updated, honest reason:

> Not "the others collapse on wheat" (false under ALIKED), but **"the others now register fine yet add no
> measurable accuracy, drift the focal/scale, crop the image, and cost more — because phone lens distortion
> here is tiny (~0.06 px of reprojection)."** It is the robust, simple, benchmark-consistent choice, **not**
> the provably most-accurate model.

**Unrun deciding test:** `image_undistorter` always feeds 3DGS a PINHOLE model, so the real question is
whether leaving ~50 px of *corner* distortion in the images (`SIMPLE_PINHOLE`) vs removing it (`OPENCV`)
changes **3DGS PSNR at the image edges**. That's a train-and-compare A/B (ideally on Euler) — the only
thing that would move the default. For **non-vegetation** scenes (buildings, structured outdoor), `OPENCV`
is already the better choice.

## 7. Planned experiment (TODO — multi-session + 3DGS PSNR)

This re-test was **one plot, geometry-only**. Before rewriting the default's rationale with full
confidence, run the deciding experiment:

1. **Generalize across sessions.** Repeat the registration + `compare_to_agisoft` sweep
   (`SIMPLE_PINHOLE | SIMPLE_RADIAL | OPENCV`, ALIKED + exhaustive) on several sessions — e.g.
   `field_A/20250618`, `field_A/20250627`, `field_D/20250627`, plus ideally one **non-vegetation** scene
   if available (where distortion modeling should help most). Confirms the "no gain" result isn't plot-specific.
2. **The real test — 3DGS PSNR.** For each camera model, train 3DGS (resolution=1, on Euler) on its
   `images/` + `sparse/0/`, render the test split, and compute **PSNR/SSIM/LPIPS**. Because the distortion
   is **corner-only**, also compute an **edge-region metric** (e.g. PSNR on an outer-border mask) — overall
   PSNR may wash out a corner-only effect.
3. **Methodological wrinkle to handle:** each camera model produces a **different undistorted image set**
   (different crop/resolution: 4032×3024 vs 3942×2956 vs 3881×2923), so the GT test pixels differ per model
   → raw PSNR is **not directly comparable across models**. Evaluate on a **common valid region** (intersect
   the undistort crops) or otherwise normalize, or the comparison is apples-to-oranges.

Decision rule: keep `SIMPLE_PINHOLE` unless `OPENCV` shows a real edge-PSNR win that survives the
common-region normalization across multiple sessions.

## 8. Files

- Test driver: `scratchpad/camtest/run_camtest.sh` (one-off; symlinks real data, writes only to scratch).
- Tool: `colmap` 4.1.0.dev0 (our CUDA build), ALIKED libs at `tools/cuda12libs/`.
- Comparison: [src/preprocessing/compare_to_agisoft.py](../src/preprocessing/compare_to_agisoft.py).
- Default + comment: [configs/preprocessing/colmap.yaml](../configs/preprocessing/colmap.yaml) `camera`.
- Related: [PHONE_SFM_FRONTEND.md](PHONE_SFM_FRONTEND.md), [PHONE_SFM_POSE_ACCURACY.md](PHONE_SFM_POSE_ACCURACY.md), [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md).
