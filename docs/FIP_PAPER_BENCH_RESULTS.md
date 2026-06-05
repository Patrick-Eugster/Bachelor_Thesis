# FIP Plots — 30k Iteration Benchmark vs. Wheat3DGS Paper

Train / render / metrics on the 7 FIP plots (461–467) at 30,000 training iterations, to compare against the original Wheat3DGS paper (which used splatfacto/gsplat). All runs done on Euler with `vanilla_3dgs.yaml` defaults at `resolution=1`.

This page tracks **three rounds of results plus a Round-3 post-mortem** of the +pp jobs that turned out worse than baseline. Skip to "Current status" at the bottom for what's actually trustworthy right now.

---

## Round 1 — Initial `paper_bench_30k` run

First clean run of all 7 plots. plot_461 was already trained as `initial_30k_iterations` from earlier work; the other 6 were trained fresh under `paper_bench_30k`.

| Plot | Experiment | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| plot_461 | `initial_30k_iterations` (older run) | **18.55** | **0.546** | **0.387** |
| plot_461 | `paper_bench_30k` (fresh run) | 22.44 | 0.655 | 0.342 |
| plot_462 | `paper_bench_30k` | 24.42 | 0.734 | 0.311 |
| plot_463 | `paper_bench_30k` | 23.69 | 0.742 | 0.298 |
| plot_464 | `paper_bench_30k` | 23.41 | 0.734 | 0.327 |
| plot_465 | `paper_bench_30k` | 24.83 | 0.777 | 0.274 |
| plot_466 | `paper_bench_30k` | 24.98 | 0.765 | 0.275 |
| plot_467 | `paper_bench_30k` | 24.90 | 0.777 | 0.285 |
| **AVG (n=7, paper_bench_30k only)** | | **24.09** | **0.741** | **0.302** |

**plot_461 puzzle:** the two 30k runs of the same plot disagreed by +3.89 PSNR. That's too large for training noise — it had to be a code/config difference between runs. Investigation below.

---

## Investigation — Why plot_461 disagreed by +3.89 PSNR

Compared the saved configs and the test-set sizes between both runs:

| Run | train | test | Split rule actually used at render time |
|---|---:|---:|---|
| `initial_30k_iterations` (older) | 30 | 6 | FIP rule: `cam_11`, `cam_12` from each prefix group held out |
| `paper_bench_30k` (newer) | 31 | 5 | llffhold=8: every 8th sorted image held out |

Two different test sets → two different PSNR numbers. Root cause was an **eval-split regression** in [`src/gaussians/scene/dataset_readers.py`](../src/gaussians/scene/dataset_readers.py): the "is this FIP?" check had been written as `image_name.startswith('cam_')`, but actual FIP filenames are `FPWW036_SR0461_FIP2_cam_11` — the `_cam_` lives in the middle, not at the start. So the predicate always returned `False` for FIP data, and every FIP plot was silently routed into the llffhold=8 fallback branch (intended for phone data).

**Fix:** replaced the predicate with `re.search(r'_cam_\d+$', image_name)`. Verified empirically:
- FIP plot_461: 36/36 names match → FIP branch → 30 train / 6 test (cam_11+cam_12 from 3 prefix groups) ✓
- Phone COLMAP `IMG_<timestamp>`: 0/119 match → llffhold branch ✓
- Phone Agisoft `IMG_<timestamp>_<seq>` (incl. `_115`, `_118`): 0/119 match → llffhold branch ✓

See [`docs/CHANGES.md`](CHANGES.md) for the per-file diff.

---

## Round 2 — Re-render + metrics with fixed predicate (FAILED — test contamination)

After fixing the predicate, re-rendered + re-computed metrics on the **already-trained** Round-1 models without retraining. The intent was a quick check; the result turned out to be untrustworthy.

| Plot | Round 1 (wrong llffhold split) | Round 2 (re-render only on cam_11+cam_12 split) | Δ |
|---|---:|---:|---:|
| plot_461 | 22.44 | 25.14 | +2.70 |
| plot_462 | 24.42 | 27.10 | +2.68 |
| plot_463 | 23.69 | 26.32 | +2.63 |
| plot_464 | 23.41 | 25.17 | +1.76 |
| plot_465 | 24.83 | 28.87 | +4.04 |
| plot_466 | 24.98 | 29.13 | +4.15 |
| plot_467 | 24.90 | 27.83 | +2.93 |
| **AVG (n=7)** | 24.09 | **27.08** | **+2.99** |

PSNR jumped **+3 dB on average** — but in the **wrong direction**. Held-out cam_11/cam_12 should be *harder* than interpolated llffhold views, not easier.

**Root cause — test contamination:** the Round 1 models were trained while the predicate was broken, so during training they used the **llffhold split** — which put `cam_11` and `cam_12` into the **train** set. When we then re-rendered with the fixed predicate, the new test set was `cam_11`+`cam_12` — images the model had already memorized during training. PSNR reflects memorization, not generalization. Numbers in Round 2 must be discarded.

**Sanity check that supports this reading:** the only honest historical number is `plot_461 / initial_30k_iterations` = **18.55 PSNR**. That model was trained much earlier, before the cam_-prefix predicate was added, when the cam_idx>10 rule actually worked on both train and test. Both sides of that run were on the correct held-out split. 18.55 is the only Round-1/2 number that hasn't been contaminated by the predicate bug.

---

## Round 3 — Full retrain with the fix (done; +pp results were worse than baseline)

Two SLURM jobs ran in parallel on Euler:

| Job | `experiment_name` | `use_principal_point` | Purpose |
|---|---|---|---|
| Baseline | `paper_bench_30k` | `false` (default) | Honest "vanilla 3DGS, no pixel-shift fix" number for the paper comparison |
| Pixel-shift fix | `paper_bench_30k_pp` | `true` (asymmetric frustum, **broken — see post-mortem below**) | Intended: honor COLMAP cx/cy ([PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md) Option 1). Actual: doubled the X shift instead of cancelling it. |

| Plot | PSNR baseline | PSNR +pp (broken) | ΔPSNR | SSIM base | SSIM +pp | LPIPS base | LPIPS +pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 461 | 18.54 | 16.75 | **−1.79** | 0.546 | 0.528 | 0.387 | 0.451 |
| 462 | 20.00 | 18.02 | **−1.97** | 0.627 | 0.602 | 0.351 | 0.428 |
| 463 | 19.66 | 17.48 | **−2.18** | 0.620 | 0.579 | 0.326 | 0.410 |
| 464 | 19.37 | 18.52 | **−0.85** | 0.633 | 0.622 | 0.358 | 0.386 |
| 465 | 23.29 | 22.09 | **−1.20** | 0.729 | 0.688 | 0.259 | 0.281 |
| 466 | 21.62 | 19.24 | **−2.37** | 0.661 | 0.604 | 0.285 | 0.368 |
| 467 | 20.13 | 18.00 | **−2.13** | 0.637 | 0.597 | 0.310 | 0.394 |
| **AVG** | **20.37** | **18.59** | **−1.78** | **0.636** | **0.603** | **0.325** | **0.388** |

Three things to note:

1. **Baseline (20.37 PSNR avg) is the first trustworthy multi-plot number** in this benchmark. plot_461 baseline = 18.54 ≈ the old `initial_30k_iterations` = 18.55, confirming the eval-split fix + clean retrain are consistent.
2. **+pp uniformly degrades metrics by ~1.8 dB.** That's the opposite of what an asymmetric-frustum cx/cy correction should do — the smoking gun for a bug in the fix itself.
3. The "splatfacto vs vanilla" gap claimed by the supervisor is largely a real gap (not all explained by a pixel shift) — see post-mortem.

---

## Post-mortem — why +pp got worse, not better

FFT phase correlation between each baseline render and its GT (first test image per plot):

| Plot | baseline (dy, dx) | +pp (dy, dx) |
|---|---|---|
| 461 | (0, **+12**) | (0, **+24**) |
| 462 | (−2, **+10**) | (0, **+20**) |
| 463 | (0, **+8**) | (0, **+15**) |
| 464 | (+7, **+7**) | (0, **+13**) |
| 465 | (−4, **+2**) | (0, **+5**) |
| 466 | (−1, **+6**) | (0, **+12**) |
| 467 | (+2, **+7**) | (0, **+14**) |

Two findings collapsed out of this:

- **The actual baseline rendered-vs-GT shift is small (2–12 px), not 50–91 px** as the `Δcx = cx − W/2` arithmetic in cameras.txt naively suggested. Reason: vanilla 3DGS keeps camera poses fixed but optimizes Gaussian positions freely, so 30k iterations absorb most of the principal-point bias into 3D scene positions. Matches the supervisor's manual visual estimate of ~5 px.
- **The +pp X shift is almost exactly 2× the baseline X shift; the +pp Y shift is always 0.** That asymmetry is a sign-flip bug — the X-axis derivation of the asymmetric frustum in `getProjectionMatrix` used the OpenGL `(right+left)/(right-left)` convention, but the 3DGS rasterizer uses `z_sign = +1` (not OpenGL's `−1`), requiring the opposite sign on `P[0,2]`. The Y formula happened to be correct because the camera Y-axis is also flipped (COLMAP convention y-down) and the two flips cancel.

Net effect: instead of cancelling the baseline ~7 px shift, the fix added another ~7 px in the same direction (doubled it). Full diagnosis in [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md).

**The sign-flip bug is now fixed in [graphics_utils.py](../src/gaussians/utils/graphics_utils.py).** The `paper_bench_30k_pp` results above were produced **before** the sign fix and should not be cited; a re-run as `paper_bench_30k_pp_signfix` is needed for the corrected +pp numbers.

---

## Round 4 — Re-run +pp after the sign-flip fix (THESIS HEADLINE)

After the X-axis sign-flip in `getProjectionMatrix` was identified and corrected, the same SLURM job was re-submitted. The script still wrote to `experiment_name=paper_bench_30k_pp`, so the Round 4 sign-fixed results **overwrote** the Round 3 broken +pp results on disk. The numbers below are the Round 4 (sign-fixed) measurements; the Round 3 broken +pp numbers above are kept in the post-mortem section for the record.

| Plot | PSNR baseline | PSNR Round-3 broken +pp | **PSNR Round-4 signfix +pp** | Δ baseline → R4 | SSIM R4 | LPIPS R4 |
|---|---:|---:|---:|---:|---:|---:|
| 461 | 18.54 | 16.75 | **25.78** | **+7.24** | 0.830 | 0.215 |
| 462 | 20.00 | 18.02 | **29.51** | **+9.51** | 0.901 | 0.189 |
| 463 | 19.66 | 17.48 | **27.50** | **+7.84** | 0.882 | 0.195 |
| 464 | 19.37 | 18.52 | **27.20** | **+7.83** | 0.874 | 0.222 |
| 465 | 23.29 | 22.09 | **28.17** | **+4.88** | 0.880 | 0.203 |
| 466 | 21.62 | 19.24 | **29.20** | **+7.58** | 0.891 | 0.184 |
| 467 | 20.13 | 18.00 | **29.80** | **+9.67** | 0.912 | 0.178 |
| **AVG** | **20.37** | **18.59** | **28.17** | **+7.79** | **0.881** | **0.198** |

SSIM avg: 0.636 → **0.881** (+0.24). LPIPS avg: 0.325 → **0.198** (−0.13). Every metric, every plot, big swing.

**Verification by FFT phase correlation** on the same first-test-image-per-plot used in the Round-3 post-mortem:

| Plot | baseline (dy, dx) | broken +pp (dy, dx) | **signfix +pp (dy, dx)** |
|---|---|---|---|
| 461 | (0, +12) | (0, +24) | **(0, 0)** |
| 462 | (−2, +10) | (0, +20) | **(0, 0)** |
| 463 | (0, +8) | (0, +15) | **(0, 0)** |
| 464 | (+7, +7) | (0, +13) | **(0, 0)** |
| 465 | (−4, +2) | (0, +5) | **(0, 0)** |
| 466 | (−1, +6) | (0, +12) | **(0, 0)** |
| 467 | (+2, +7) | (0, +14) | **(0, 0)** |

**Zero shift on every plot.** The asymmetric-frustum formula in `getProjectionMatrix` now reproduces the COLMAP principal point exactly, and the renderer's output pixel-aligns with the GT image. Combined with the metric jump, the principal-point projection inaccuracy was clearly the dominant gap between vanilla 3DGS and the splatfacto baseline on FIP data.

**Why +7.8 dB and not the sub-dB predicted in Round 3?** The Round-3 post-mortem reasoning ("residual pixel shift is only ~7 px → fix can only buy fractions of a dB") was wrong. The principal-point bias isn't only spent on a global pixel shift — vanilla 3DGS absorbs it by moving individual 3D Gaussians to geometrically-wrong positions, which then bind wrong colors/textures to wrong pixels and hurts SSIM/LPIPS independently of any global alignment. When `use_principal_point=true` lets the projection sit where the camera actually points, every Gaussian can land in its correct place and the entire scene reconstructs much more faithfully — not just a global re-alignment.

---

## Current status

- ✅ Eval-split regression understood and fixed in [dataset_readers.py](../src/gaussians/scene/dataset_readers.py) (the `_cam_\d+$` regex).
- ✅ Pixel-shift bug Option 1 implemented as an opt-in flag, and the X-axis sign-flip in the asymmetric frustum now corrected (see [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md) and [CHANGES.md](CHANGES.md)). Default off → no behavior change for unaware users.
- ✅ Round 3 **baseline** finished — `paper_bench_30k` avg PSNR = 20.37, the first trustworthy multi-plot number. plot_461 = 18.54 ≈ `initial_30k_iterations` 18.55 confirms cleanliness.
- ✅ Round 4 **signfix +pp** finished — `paper_bench_30k_pp` (overwrote the Round-3 broken folder) avg PSNR = **28.17** (+7.79 dB over baseline), avg SSIM **0.881**, avg LPIPS **0.198**. FFT phase correlation confirms zero render-vs-GT pixel shift on all 7 plots.
- ⛔ Round 3 **+pp** (sign-flipped frustum) is degraded (avg PSNR 18.59) — kept in the "Round 3" + "Post-mortem" sections above as the historical record of the bug. The Round-4 results superseded it on disk.
- ⛔ Round 1 (24.09 PSNR) and Round 2 (27.08 PSNR) numbers still should not be cited — Round 1 trained with the broken predicate, Round 2 was test-contaminated.

## Next steps

1. **Compare Round-4 signfix +pp (28.17 PSNR avg) against the original Wheat3DGS paper's splatfacto numbers** — this is the headline thesis comparison. If close, the principal-point fix is the headline finding.
2. If a noticeable residual gap to splatfacto remains, the next move is Option 2 (gsplat) — see [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md) "Fix options" §. The kernel-level cx/cy handling could close any remaining tile-binning artifacts at image borders.
3. Re-run segmentation + downstream phenotyping with `use_principal_point=true` and compare against baseline — the same Gaussian-position correction that helped PSNR should also tighten 3D head IDs / phenotyping accuracy.
