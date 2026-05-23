# FIP Plots — 30k Iteration Benchmark vs. Wheat3DGS Paper

Train / render / metrics on the 7 FIP plots (461–467) at 30,000 training iterations, to compare against the original Wheat3DGS paper (which used splatfacto/gsplat). All runs done on Euler with `vanilla_3dgs.yaml` defaults at `resolution=1`.

## Per-plot results

| Plot | Experiment | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| plot_461 | `initial_30k_iterations` (older run) | **18.55** | **0.546** | **0.387** |
| plot_462 | `paper_bench_30k` | 24.42 | 0.734 | 0.311 |
| plot_463 | `paper_bench_30k` | 23.69 | 0.742 | 0.298 |
| plot_464 | `paper_bench_30k` | 23.41 | 0.734 | 0.327 |
| plot_465 | `paper_bench_30k` | 24.83 | 0.777 | 0.274 |
| plot_466 | `paper_bench_30k` | 24.98 | 0.765 | 0.275 |
| plot_467 | `paper_bench_30k` | 24.90 | 0.777 | 0.285 |
| **Average (n=7)** | | **23.54** | **0.725** | **0.308** |
| **Average without 461 (n=6)** | | **24.37** | **0.755** | **0.295** |

## Observations

- **plot_461 is a clear outlier**: PSNR ~6 dB lower than the others, SSIM ~0.19 lower, LPIPS noticeably worse. It was trained earlier with a different experiment name (`initial_30k_iterations`) and possibly different config / code state. A clean rerun with `paper_bench_30k` is queued so the comparison is apples-to-apples.
- **Plots 462–467 are tightly clustered** (PSNR 23.4–25.0, SSIM 0.73–0.78). That tightness is reassuring — it's the same pipeline applied to seven scenes of the same nature, and the spread is small. It also means the n=6 average is a robust headline number, not a single lucky/unlucky plot.
- **All numbers are still affected by the principal-point bug** documented in [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md). Every FIP plot has a non-trivial cx offset (−20 to −91 px) that the vanilla rasterizer silently discards, so renders are misaligned with GT by tens of pixels. The clustered 23–25 PSNR is the *with-shift* ceiling. Once the bug is fixed (gsplat / asymmetric frustum / patched rasterizer), expect a meaningful jump.
- **Paper comparison**: the original Wheat3DGS paper reports higher metrics on FIP plots because splatfacto/gsplat plumbs cx/cy correctly. So the gap between our numbers and the paper's numbers is at least partly explained by the pixel-shift bug — not by undertraining, scene difficulty, or a worse 3D model.

## Next steps

1. **Rerun plot_461** with `experiment_name=paper_bench_30k` so all 7 plots use the identical pipeline + naming. Script ready, queued on Euler.
2. **Re-average** once plot_461 finishes — that's the headline n=7 number for the paper comparison.
3. **Pick a fix path for the pixel-shift bug** (see [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md) §"What fixing it would take" for the three options) and rerun plot_461 (or all seven) to quantify the gain. That delta is the most thesis-relevant single number: "vanilla 3DGS without cx/cy: PSNR X.XX; with cx/cy: PSNR Y.YY; paper splatfacto: PSNR Z.ZZ."
