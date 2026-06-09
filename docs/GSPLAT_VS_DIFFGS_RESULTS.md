# gsplat vs diff-gaussian — full-pipeline benchmark (7 FIP plots)

**Question:** is the gsplat render engine (branch `gsplat-switch`) a safe replacement for the
original `diff-gaussian-rasterization`, with no loss of render quality or segmentation accuracy,
and is the residual scalar-tanfov ("tan") approximation actually negligible?

**Answer: yes on all counts — gsplat matches diff-gaussian on quality + segmentation and trains
~1.77× faster.**

> ⚠️ **Scope: FIP data only (7 plots).** This benchmark has **not** been run on phone data yet.
> gsplat-vs-diffgs on phone (quality + segmentation + timing) is a separate **TODO** before any
> general "gsplat is better for the project" claim — phone has different image characteristics
> (resolution, sparser SfM, more elongated heads) that could shift the numbers.

**Runs (Euler, 2026-06-08/09, all 7 FIP plots, 30k iters, `use_principal_point=true`):**
- `test_diffgs_full` — diff-gaussian engine (`WHEAT_RENDERER=diffgs`)
- `test_gsplat_full` — gsplat engine (default) + Test A (`compare_renderers.py`)

**Script:** [`src/analysis/compare_gsplat_runs.py`](../src/analysis/compare_gsplat_runs.py).
**Raw data:** [`analysis_results/gsplat_vs_diffgs.json`](analysis_results/gsplat_vs_diffgs.json).
Background: [`GSPLAT_PORT.md`](GSPLAT_PORT.md), [`PIXEL_SHIFT_BUG.md`](PIXEL_SHIFT_BUG.md).

---

## 1. Render quality (test split, 30k)

| plot | diffgs PSNR/SSIM/LPIPS | gsplat PSNR/SSIM/LPIPS | ΔPSNR |
|---|---|---|--:|
| plot_461 | 25.79 / 0.830 / 0.213 | 25.76 / 0.829 / 0.216 | −0.04 |
| plot_462 | 29.51 / 0.902 / 0.188 | 29.54 / 0.902 / 0.188 | +0.02 |
| plot_463 | 27.45 / 0.882 / 0.195 | 27.50 / 0.882 / 0.196 | +0.06 |
| plot_464 | 27.19 / 0.874 / 0.222 | 27.22 / 0.874 / 0.222 | +0.02 |
| plot_465 | 28.17 / 0.880 / 0.203 | 28.21 / 0.881 / 0.204 | +0.04 |
| plot_466 | 29.20 / 0.891 / 0.183 | 29.16 / 0.891 / 0.184 | −0.04 |
| plot_467 | 29.85 / 0.912 / 0.179 | 29.84 / 0.912 / 0.179 | −0.02 |
| **AVG** | **28.17 / 0.881 / 0.198** | **28.17 / 0.881 / 0.198** | **+0.01** |

→ Per-plot PSNR differences are within **±0.06 dB** (noise); SSIM/LPIPS identical. **No quality regression.**

---

## 2. Test A — engine agreement (the "tan-residual" check)

`compare_renderers.py` renders every view with **both** engines on the *same* trained gsplat model
and compares them. The `alpha_IoU` (alpha>0.5 masks = the `pred_seg` the segmentation matcher uses)
directly bounds the residual scalar-tanfov shape distortion.

| plot | alpha_IoU | RGB PSNR (engine-vs-engine) | disagree px |
|---|--:|--:|--:|
| plot_461 | 0.999992 | 64.74 | 96 |
| plot_462 | 0.999999 | 64.56 | 7 |
| plot_463 | 0.999999 | 64.59 | 13 |
| plot_464 | 1.000000 | 65.92 | 4 |
| plot_465 | 1.000000 | 66.20 | 4 |
| plot_466 | 0.999998 | 65.56 | 30 |
| plot_467 | 0.999999 | 64.73 | 7 |
| **AVG** | **0.999998** | **65.19** | — |

→ The two engines are **essentially pixel-identical** (alpha-IoU 0.999998, 65 dB). **The residual
tan approximation is negligible** — this closes the open question from
[`DENSIFICATION_OPTIONS.md`](DENSIFICATION_OPTIONS.md)/the flashsplat tan-residual discussion: there
is no measurable shape distortion to worry about, so porting flashsplat off diff-gaussian is **not**
needed for correctness.

---

## 3. Segmentation accuracy vs ground truth (eval_2d, one labelled view per plot)

| plot | diffgs IoU / F1  (pred/gt heads) | gsplat IoU / F1  (pred/gt heads) | ΔIoU |
|---|---|---|--:|
| plot_461 | 0.532 / 0.694  (231/218) | 0.518 / 0.682  (220/218) | −0.014 |
| plot_462 | 0.696 / 0.821  (259/229) | 0.695 / 0.820  (264/229) | −0.001 |
| plot_463 | 0.495 / 0.662  (135/141) | 0.509 / 0.674  (132/141) | +0.014 |
| plot_464 | 0.687 / 0.814  (240/196) | 0.685 / 0.813  (239/196) | −0.002 |
| plot_465 | 0.684 / 0.812  (290/233) | 0.693 / 0.819  (285/233) | +0.009 |
| plot_466 | 0.303 / 0.465  (102/153) | 0.298 / 0.460  ( 98/153) | −0.005 |
| plot_467 | 0.723 / 0.839  (302/216) | 0.717 / 0.835  (309/216) | −0.006 |
| **AVG** | **0.588 / 0.730** | **0.588 / 0.729** | **−0.001** |

→ Segmentation accuracy is **identical** (ΔIoU −0.001, ΔF1 −0.001); head counts track within a few.
The concern that gsplat's slightly different blob shapes could mis-match heads is **not borne out**.

### 3D heads assigned (segmentation_3d/run_1/results.csv)

| plot | diffgs #heads / matches | gsplat #heads / matches | Δheads |
|---|---|---|--:|
| plot_461 | 511 / 7838 | 493 / 7546 | −18 |
| plot_462 | 560 / 8448 | 552 / 8333 | −8 |
| plot_463 | 570 / 9006 | 580 / 9104 | +10 |
| plot_464 | 582 / 8431 | 583 / 8525 | +1 |
| plot_465 | 512 / 7383 | 504 / 7472 | −8 |
| plot_466 | 360 / 4798 | 361 / 4644 | +1 |
| plot_467 | 643 / 9651 | 653 / 9527 | +10 |

→ Head counts agree within ±18 (mostly single digits) — within run-to-run matching noise.

---

## 4. Timing — gsplat trains ~1.77× faster

Per-plot 30k-iteration training wall-time (both jobs on **`rtx_4090`**, from the SLURM `.out` logs):

| plot | diffgs train | gsplat train | speedup |
|---|--:|--:|--:|
| plot_461 | 1:34:02 | 54:45 | 1.72× |
| plot_462 | 1:27:32 | 48:54 | 1.79× |
| plot_463 | 1:30:03 | 51:00 | 1.77× |
| plot_464 | 1:27:51 | 49:59 | 1.76× |
| plot_465 | 1:28:21 | 49:28 | 1.79× |
| plot_466 | 1:28:53 | 49:59 | 1.78× |
| plot_467 | 1:27:05 | 48:57 | 1.78× |
| **AVG** | **1:29:07** | **50:26** | **1.77×** |

→ **gsplat trains ~1.77× faster than diff-gaussian at identical quality** — ~89 min → ~50 min per
plot, dead consistent across all 7 plots (1.72–1.79×). For the full 7-plot benchmark that is
**~10.4 h → ~5.9 h**.

Notes: both jobs requested the same GPU (`rtx_4090`), so this is a like-for-like render-engine
comparison, not a hardware artifact. **Segmentation** runs on flashsplat for both engines, so its
timing is not a render-engine signal and is excluded; the render step is dominated by the training
loop measured above.

---

## 5. Downsides of gsplat (none in results; engineering only)

- **No result downside:** quality, segmentation accuracy, and head counts are all equal; the
  pixel-shift fix is now **native** (gsplat takes the full intrinsic matrix `K`), removing the
  asymmetric-frustum hack.
- **New dependency** `gsplat==1.5.3` — the Euler env must stay synced (`pip install -e .`); a stale
  env silently breaks pipeline steps (see [`euler_setup.md`](euler_setup.md)).
- **More moving parts** in `render()`: the engine switch, means2d→NDC gradient rescale, per-axis
  radii collapse, viewmat transpose — extra surface to maintain.
- **Two engines coexist:** train/render/eval on gsplat, segmentation still on flashsplat — the
  codebase grew rather than simplified.
- **First-run JIT compile** of gsplat CUDA kernels per GPU (one-time).
- **Upside beyond parity:** **~1.77× faster training** at identical quality (Section 4), and unlocks
  gsplat's **MCMC / AbsGrad** densification strategies for the next experiment (see
  [`DENSIFICATION_OPTIONS.md`](DENSIFICATION_OPTIONS.md)).

---

## 6. Conclusion

**gsplat is a strict win over diff-gaussian** — quality-neutral (ΔPSNR +0.01 dB),
segmentation-neutral (ΔIoU −0.001), residual tan approximation **negligible** (engine alpha-IoU
0.999998), **and ~1.77× faster to train** (~89 min → ~50 min per plot, Section 4). The
`gsplat-switch` branch should be made the default. The only costs are a new dependency and added
code complexity — both outweighed by the near-2× training speedup at identical results.

> ⚠️ **Process note (why this benchmark was almost incomplete):** the Euler runs produced **no**
> `metrics_2d.json` because `eval_seg_2d.py` crashed at import (`scikit-image` missing in the Euler
> env, though it is declared in `pyproject.toml`), and `run_reconstruction.py` continues past a
> failed step so the SLURM job still reported success. The seg metrics in Section 3 were regenerated
> by re-running step 6b **locally** (GT masks + predictions both present locally), overriding `-s`/`-m`
> because the saved `cfg_args` holds the Euler path. Fix: keep the Euler env synced with
> `pip install -e .` (see [`euler_setup.md`](euler_setup.md)).
