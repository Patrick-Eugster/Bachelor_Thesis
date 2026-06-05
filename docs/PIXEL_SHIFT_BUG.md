# Pixel-Shift Bug in train / render / metrics Pipeline

Investigation into the supervisor-reported "pixel shifting" issue that makes our 3DGS metrics worse than the original Wheat3DGS paper's (splatfacto). This doc records the **full** journey: an initial wrong hypothesis (90 px shift), an opt-in fix that made metrics worse, an FFT phase-correlation re-measurement that exposed both the actual (much smaller) shift and a sign-flip bug in the fix, and the corrected implementation.

## TL;DR

1. Vanilla 3DGS in this repo *does* drop COLMAP/Agisoft's principal point `(cx, cy)` and forces the projection to assume the optical axis is at `(W/2, H/2)`.
2. The **real** rendered-vs-GT misalignment from this is small — ~2 to 12 px on the FIP plots, not the 50–91 px that naive `Δcx = cx − W/2` arithmetic from cameras.txt suggests. Reason: vanilla 3DGS keeps camera poses fixed but optimizes Gaussian positions freely, so 30k iterations of training absorb most of the principal-point bias into the 3D scene itself. What survives is what FFT phase-correlation actually picks up.
3. The original opt-in fix (`use_principal_point=true`) was built with the OpenGL `(right+left)/(right-left)` convention, but the 3DGS rasterizer uses `z_sign=+1` (not OpenGL's `-1`), which requires the **opposite sign** on `P[0,2]`. The Y formula was accidentally correct because the camera also uses Y-down (COLMAP convention) — the two negations cancel for Y but not for X. Empirically the broken fix **doubled the X shift instead of cancelling it**, dropping average PSNR by ~1.8 dB.
4. Sign flip is now fixed in `getProjectionMatrix`. Re-running `paper_bench_30k_pp` is required to get correct +pp numbers; expected gain over baseline is small (sub-dB, since the underlying shift is small).

## The original bug: principal point silently dropped

**Where:** [src/gaussians/scene/dataset_readers.py:89-99](../src/gaussians/scene/dataset_readers.py#L89-L99) (pre-fix).

COLMAP intrinsics contain 4 numbers (`fx, fy, cx, cy` for PINHOLE; `f, cx, cy` for SIMPLE_PINHOLE). The vanilla 3DGS reader only consumed the focal length(s). The full chain that depended on it:

- `CameraInfo` originally had no `cx`/`cy` fields (now extended; see "Fix" section below)
- `Camera` stores `FoVx, FoVy` and builds its projection matrix from those
- `getProjectionMatrix` in [graphics_utils.py](../src/gaussians/utils/graphics_utils.py) builds a **symmetric frustum** (`left=-right`, `bottom=-top`), forcing the principal point to image center
- The CUDA rasterizer takes only `tanfovx, tanfovy, projmatrix` — no cx/cy slot in [gaussian_renderer/__init__.py](../src/gaussians/gaussian_renderer/__init__.py)

So every rendered pixel is projected as if the camera looked straight through `(W/2, H/2)`, even when the COLMAP intrinsic says the optical axis is elsewhere.

## What we got wrong the first time: magnitude

The first version of this doc tabulated `Δcx = cx − W/2` and `Δcy = cy − H/2` from each plot's cameras.txt and called those the misalignment:

| Plot | W×H | cx | cy | Δcx (px) | Δcy (px) |
|---|---|---|---|---|---|
| 461 | 4095×2996 | 1956 | 1480 | −91.5 | −18.0 |
| 462 | 4095×2995 | 1969 | 1513 | −78.5 | +15.5 |
| 463 | 4094×2998 | 1987 | 1499 | −60.0 | +0.0 |
| 464 | 4093×2997 | 1996 | 1438 | −50.5 | −60.5 |
| 465 | 4095×2998 | 2027 | 1524 | −20.5 | +25.0 |
| 466 | 4096×2996 | 2001 | 1506 | −47.0 | +8.0 |
| 467 | 4094×2996 | 1991 | 1483 | −56.0 | −15.0 |

That arithmetic is real but **doesn't equal the rendered-vs-GT pixel misalignment** in a vanilla 3DGS trained model. Vanilla 3DGS doesn't optimize camera poses — they're fixed — but it does optimize every Gaussian's 3D position freely. Over 30k iterations, the optimizer moves Gaussians slightly so that the symmetric-frustum projection of those moved Gaussians lands close to the GT. The principal-point bias is largely absorbed into the scene itself.

## What the actual shift is: FFT phase correlation

Measured directly on each plot's first test image with FFT phase correlation (cross-correlation peak between baseline render and its GT), on the trained Round-3 baseline models:

| Plot | baseline (dy, dx) | broken-fix +pp (dy, dx) |
|---|---|---|
| 461 | (0, **+12**) | (0, **+24**) |
| 462 | (−2, **+10**) | (0, **+20**) |
| 463 | (0, **+8**) | (0, **+15**) |
| 464 | (+7, **+7**) | (0, **+13**) |
| 465 | (−4, **+2**) | (0, **+5**) |
| 466 | (−1, **+6**) | (0, **+12**) |
| 467 | (+2, **+7**) | (0, **+14**) |

Baseline residual shift is **2–12 px** in x (median ~7), and ±7 px in y. Matches the manual visual estimate of "at most ~5 px." Not 90 px.

## The sign-flip bug in the first fix attempt

Two empirical clues from the table above:

- The `+pp` X shift is **almost exactly 2× the baseline X shift** (12→24, 10→20, 8→15, …).
- The `+pp` Y shift is **always 0** (the y-axis correction worked).

That asymmetry is the smoking gun. Re-deriving from scratch for the 3DGS rasterizer convention:

The 3DGS rasterizer uses **z_sign = +1** (P[3,2]=+1, w_clip = +z_cam, camera looks down +Z, COLMAP convention) and the camera **Y axis is down** (COLMAP convention, matches image Y). Combined with the standard NDC-to-pixel mapping `pixel = (ndc+1)·S/2`, the principal-point pixel `(cx, cy)` lands at NDC:

- `ndc_x = 2·cx/W − 1`
- `ndc_y = 2·cy/H − 1`

For a point on the optical axis (`x_cam = y_cam = 0`), `ndc_x = P[0,2]` and `ndc_y = P[1,2]`. So the correct projection-matrix entries are:

- **`P[0,2] = 2·cx/W − 1`**
- **`P[1,2] = 2·cy/H − 1`**

The first (broken) `getProjectionMatrix` used the standard OpenGL formulae `P[0,2] = (right+left)/(right-left)` with `right = (W-cx)·n/fx`, `left = -cx·n/fx`. That gives `P[0,2] = (W − 2cx)/W = −(2cx/W − 1)` — **the negative of what's required**. For Y the OpenGL formula gave `(2cy − H)/H = 2cy/H − 1`, which happens to match — because the camera Y-axis is also flipped relative to OpenGL, and the two flips cancel.

Net effect of the broken fix: for plot 461 (cx-effective ≈ −12 px in image space), the symmetric baseline already had +12 px of residual shift after training. The broken fix shifted by an additional 12 px in the **same** direction instead of cancelling, producing the observed +24 px shift in the `+pp` run.

## Why the image-space shift is ~12 px, not ~91 px

Two reasons combined:

1. **Gaussian-position optimization absorbs most of the principal-point bias** during the 30k-iteration train. Camera poses are fixed in vanilla 3DGS, but the 3D Gaussians themselves move freely.
2. Even before absorption, the literal "every rendered pixel projects through (W/2, H/2) instead of (cx, cy)" claim assumes the in-cameras.txt values are the literal pixel-space principal point. They are, but for a typical FIP plot the residual after pose-fixed Gaussian-position optimization is much smaller than the raw `Δcx`.

## The fix (corrected)

Asymmetric frustum in [src/gaussians/utils/graphics_utils.py](../src/gaussians/utils/graphics_utils.py) `getProjectionMatrix`, with the X branch **rewritten so the sign matches the 3DGS rasterizer convention**:

```python
# X (sign-corrected for 3DGS convention)
left   = -(width - cx) * znear / fx   # was: -cx * znear / fx           ← BROKEN
right  =  cx           * znear / fx   # was: (width - cx) * znear / fx  ← BROKEN
# Y (unchanged — already correct because camera Y is down)
top    =  cy            * znear / fy
bottom = -(height - cy) * znear / fy
```

Sanity check at `cx = W/2` (symmetric edge case):
- `left = -W/2·n/fx = -tanHalfFovX·n` ✓
- `right = W/2·n/fx = +tanHalfFovX·n` ✓
- `P[0,2] = (right+left)/(right-left) = 0` ✓
- `P[0,0] = 2·n/(right-left) = 2fx/W = 1/tanHalfFovX` ✓ (same as old symmetric branch)

For general cx:
- `P[0,2] = (2·cx − W)/W = 2·cx/W − 1` ✓ correct sign
- `P[0,0] = 2·n/(right-left) = 2·n / (W·n/fx) = 2·fx/W` ✓ unchanged

## Splatfacto vs gsplat vs diff-gaussian-rasterization

Splatfacto (Nerfstudio) runs on **gsplat**, which takes the full intrinsic matrix `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]` as input and bakes cx/cy into the projection kernel directly. Vanilla `diff-gaussian-rasterization` (what this repo uses) takes only `tanfovx, tanfovy, projmatrix`, so cx/cy can only be smuggled in via `projmatrix` (which is what Option 1 does). Both work, but only gsplat handles cx/cy *inside the kernel's tile binning / radii* — see "Known residual" below.

## Fix options (recap)

1. **Asymmetric projection matrix** — implemented (and now sign-corrected). Smuggles cx/cy through `projmatrix` so projection lands the optical axis at `(cx, cy)`. Works for both `diff-gaussian-rasterization` and `flashsplat_rasterization`. Subtle limitation: the kernel's `tanfovx/tanfovy` (scalar) tile binning still assumes symmetric frustum — negligible at our ~7 px residual offset.
2. **Switch to gsplat** for the rasterizer (drop-in `rasterization()` call that accepts `Ks` with cx/cy). Cleanest *correct* fix; tile binning, culling, Jacobian — all aware of cx/cy. Caveat: `flashsplat_rasterization` (used by segmentation_3d) is independent of gsplat and would still need its own fix.
3. **Patch diff-gaussian-rasterization CUDA** to accept cx/cy. Means rebuilding the CUDA submodule with extra args, propagating cx/cy through `GaussianRasterizationSettings`, fixing the CUDA projection. Invasive; not worth it given (1) covers our case.

## Option 1 — implemented as opt-in (December 2025; sign-corrected after Round 3)

Option 1 is **implemented in the code** behind a config flag. Default-off so existing behavior is byte-identical to vanilla 3DGS.

**Files changed:** `configs/reconstruction_seg3d/reconstruction/vanilla_3dgs.yaml` (added `use_principal_point: false`), `src/gaussians/arguments/__init__.py` (ModelParams gained the flag), `src/gaussians/scene/dataset_readers.py` (cx/cy parsed from intrinsics, CameraInfo extended), `src/gaussians/utils/camera_utils.py` (scales cx/cy with image resize), `src/gaussians/scene/cameras.py` (Camera stores cx/cy), `src/gaussians/utils/graphics_utils.py` (asymmetric frustum branch — **sign-corrected for 3DGS convention**), `src/run_reconstruction.py` (passes `--use_principal_point` to all 6 camera-using subprocess calls). Full per-file diff in [docs/CHANGES.md](CHANGES.md).

**How to enable:**
```bash
# Default (no fix — for paper baseline comparison)
python src/run_reconstruction.py plot=plot_461 run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=30000 experiment_name=paper_bench_30k

# With pixel-shift fix (post-sign-correction — needs re-run; old `paper_bench_30k_pp` is broken)
python src/run_reconstruction.py plot=plot_461 run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=30000 reconstruction.use_principal_point=true \
  experiment_name=paper_bench_30k_pp_signfix
```

Different `experiment_name` so the runs go to different folders, no overwrite risk.

**Known residual artifact:** the CUDA kernel's tile binning and culling still use `tanfovx, tanfovy` (single scalars) which assume a symmetric frustum. For our residual offsets (~7 px on 4K, <0.2% of image width) this means at most sub-pixel mis-binning at the image **borders only** — center pixels (~99% of image area) get the projection-matrix shift exactly right. If the sign-fixed Option 1 still shows weird edge artifacts, Option 2 (gsplat) is the next step.

## Recommended path forward (as planned before re-run)

1. **Re-run +pp with the sign fix** on Euler — same SLURM job as before but with the freshly-fixed `graphics_utils.py`. Use a new `experiment_name=paper_bench_30k_pp_signfix` so the broken `paper_bench_30k_pp` results stay archived for comparison.
2. **Expected delta is small** (fractions of a dB), since the underlying residual shift was only ~7 px. The bigger story for the thesis is that vanilla 3DGS already lands close to the paper's numbers once eval-split + the sign fix are both right.
3. If a noticeable gap remains vs splatfacto — switch to Option 2 (gsplat). The cleaner fix is also Blackwell-ready (helps with the RTX Pro 6000 future task in `docs/euler_setup.md`).
4. Option 3 (patch CUDA) remains the fallback if for some reason gsplat can't be adopted — likely never needed.

---

## What actually happened after the re-run — sign-fix verified, gain is enormous

The +pp re-run with the sign-fixed code was submitted. The SLURM script was *not* renamed to `paper_bench_30k_pp_signfix`, so the new outputs **overwrote** the Round-3 broken `paper_bench_30k_pp/` folder on disk. The per-plot table for the sign-fixed run is in [FIP_PAPER_BENCH_RESULTS.md](FIP_PAPER_BENCH_RESULTS.md) "Round 4" — headline numbers:

| Metric | Baseline (no fix) | Round-3 broken +pp | **Round-4 signfix +pp** |
|---|---:|---:|---:|
| PSNR avg | 20.37 | 18.59 (−1.78) | **28.17 (+7.79)** |
| SSIM avg | 0.636 | 0.603 | **0.881** |
| LPIPS avg | 0.325 | 0.388 | **0.198** |

**FFT phase correlation now reports (0, 0) shift on every plot** — exact pixel alignment.

### The "sub-dB expected" prediction was wrong — here's why

The pre-run prediction said "the underlying residual shift was only ~7 px → fix can only buy fractions of a dB." Reality: +7.8 dB average. The reasoning that produced "sub-dB" treated the residual shift as the *only* thing the fix corrects. In fact:

1. **Vanilla 3DGS absorbs the principal-point bias by moving Gaussians to geometrically wrong 3D positions** — not just by leaving a global pixel shift. Camera poses are fixed, but Gaussian positions are free.
2. Those wrong positions bind **wrong colors / textures to wrong pixels**, hurting SSIM and LPIPS independently of the global pixel alignment. PSNR catches the same effect because the wrongly-placed appearance content shows up as elevated MSE in every test view.
3. The 2–12 px residual rendered-vs-GT shift was the *easily-measurable* leftover. The much larger effect (every Gaussian sitting at a slightly biased position) wasn't visible until the fix let the scene fully reconstruct.

So the FFT-phase-correlation measurement (~7 px) **understated** the bug. The (0, 0) measurement post-fix is necessary but only part of the story — the rest is the entire 3D scene now being placed in self-consistent geometry.

### Updated recommendation

- The Round-4 signfix-+pp numbers (28.17 PSNR avg) are now the **headline thesis-relevant result** for vanilla 3DGS on FIP plots. Compare against the original Wheat3DGS paper's splatfacto numbers — if comparable, the principal-point fix is the main finding for this section of the thesis.
- Option 2 (gsplat) is no longer urgent. The only motivation left is the residual tile-binning approximation at image borders (uses single-scalar tanfovx/tanfovy assuming symmetric), but for our ~7 px effective offsets that's far below a tile boundary. Only revisit if Round-4 visually shows weird edge artifacts.
- Option 3 (patch CUDA) remains a dead-end backup — likely never needed.
- The same fix should also benefit segmentation + downstream phenotyping (same Gaussian-position correctness argument). Re-run those stages with `use_principal_point=true` and compare.
