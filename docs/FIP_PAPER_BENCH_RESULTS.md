# FIP Plots — 30k Iteration Benchmark vs. Wheat3DGS Paper

Train / render / metrics on the 7 FIP plots (461–467) at 30,000 training iterations, to compare against the original Wheat3DGS paper (which used splatfacto/gsplat). All runs done on Euler with `vanilla_3dgs.yaml` defaults at `resolution=1`.

## Per-plot results

| Plot | Experiment | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| plot_461 | `initial_30k_iterations` (older run, kept for diagnosis) | 18.55 | 0.546 | 0.387 |
| plot_461 | `paper_bench_30k` (clean rerun) | **22.44** | **0.655** | **0.342** |
| plot_462 | `paper_bench_30k` | 24.42 | 0.734 | 0.311 |
| plot_463 | `paper_bench_30k` | 23.69 | 0.742 | 0.298 |
| plot_464 | `paper_bench_30k` | 23.41 | 0.734 | 0.327 |
| plot_465 | `paper_bench_30k` | 24.83 | 0.777 | 0.274 |
| plot_466 | `paper_bench_30k` | 24.98 | 0.765 | 0.275 |
| plot_467 | `paper_bench_30k` | 24.90 | 0.777 | 0.285 |
| **Average (n=7, paper_bench_30k only)** | | **24.09** | **0.741** | **0.302** |

### plot_461 — `initial_30k_iterations` vs `paper_bench_30k` discrepancy

Two 30k runs of the **same plot** produced very different metrics:

| Metric | `initial_30k_iterations` | `paper_bench_30k` | Δ |
|---|---:|---:|---:|
| PSNR  | 18.55 | **22.44** | **+3.89** |
| SSIM  | 0.546 | **0.655** | **+0.109** |
| LPIPS | 0.387 | **0.342** | **−0.045** |

A +3.89 PSNR jump on the same plot with the same nominal pipeline is far too large to be training noise — it points at a real difference in code, config, or eval setup between the two runs. Worth investigating before drawing conclusions from the older numbers:
- Was the older run done before the eval-split fix in [dataset_readers.py:191-197](../src/gaussians/scene/dataset_readers.py#L191-L197)? A wrong train/test split (e.g. mostly-test instead of FIP's cam_11+cam_12 only) would tank PSNR.
- Were the auto-cleanup of stale renders / metrics-error-visibility fixes in place when the older run produced its `results.json`? Stale leftover renders from a different split would cause metrics to be computed on the wrong file set.
- Were training hyperparameters identical (sh_degree, densify_until_iter, opacity_cull_threshold)? Check the saved `config.yaml` inside each experiment folder.
- Did the older run actually finish all 30k iterations, or was it loaded from an early checkpoint?

The newer `paper_bench_30k` row is the trustworthy one (clean SLURM run with all current fixes applied). The older row is kept here only for diagnosis.

## Observations

- **plot_461 is still the lowest of the seven** (22.44 PSNR vs cluster of 23.4–25.0), but the gap is now ~1 PSNR below the next plot (464 at 23.41) — within plausible scene-difficulty variation, not a different-pipeline outlier like before.
- **Plots 462–467 are tightly clustered** (PSNR 23.4–25.0, SSIM 0.73–0.78). That tightness is reassuring — it's the same pipeline applied to seven scenes of the same nature, and the spread is small.
- **All numbers are still affected by the principal-point bug** documented in [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md). Every FIP plot has a non-trivial cx offset (−20 to −91 px) that the vanilla rasterizer silently discards, so renders are misaligned with GT by tens of pixels. The clustered 23–25 PSNR is the *with-shift* ceiling. Once the bug is fixed (gsplat / asymmetric frustum / patched rasterizer), expect a meaningful jump.
- **Paper comparison**: the original Wheat3DGS paper reports higher metrics on FIP plots because splatfacto/gsplat plumbs cx/cy correctly. So the gap between our numbers and the paper's numbers is at least partly explained by the pixel-shift bug — not by undertraining, scene difficulty, or a worse 3D model.

## Next steps

1. **Investigate the plot_461 discrepancy** between `initial_30k_iterations` (18.55 PSNR) and `paper_bench_30k` (22.44 PSNR) — see questions in the section above. The 3.89 dB jump is too large to be training noise; we want to know what changed so the older run's numbers can be either trusted or formally discarded.
2. **Run the asymmetric-frustum A/B** ([PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md) Option 1, already implemented as `reconstruction.use_principal_point=true`) on plot_461 (and ideally all seven). That delta is the most thesis-relevant single number: "vanilla 3DGS without cx/cy: PSNR X.XX; with cx/cy: PSNR Y.YY; paper splatfacto: PSNR Z.ZZ."
