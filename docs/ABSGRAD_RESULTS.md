# AbsGrad Densification — Results on FIP (7 plots)

**Headline:** Switching the densification criterion to **AbsGrad (AbsGS)** — gsplat's *absolute* screen-space gradient `means2d.absgrad` for the split/clone decision — improves reconstruction on **every** FIP plot (7/7) and segmentation on **6 of 7** (one near-tie loss), at **no extra training cost**. Best operating point: **gsplat + AbsGrad @ 15k iterations**.

> One-liner for the thesis: *gsplat makes training faster; AbsGrad makes the result sharper — and the thesis wants sharper (fine wheat detail: awns, edges).*

---

## TL;DR

- **Rendering (30k):** AbsGrad beats the gsplat-default control on **7/7 plots, all 3 metrics** — mean **PSNR +0.53 dB**, **SSIM +0.012**, **LPIPS −0.026** (LPIPS lower = better).
- **Segmentation (eval_2d, 7/7 plots):** AbsGrad wins/ties on **6 of 7** — mean **IoU +0.022**, **F1 +0.019** (loses only plot 467, −0.011 IoU, where extra recall tipped into over-counting). Mechanism: higher head **recall** at a small precision cost.
- **15k vs 30k:** essentially **flat** (PSNR −0.02, LPIPS −0.004 at 30k) → **use 15k**, half the training for the same quality.
- **No over-densification:** Gaussian counts stay in the vanilla range (~1.0–1.45 M), so AbsGrad@`densify_grad_threshold=0.0008` is stable.
- **The perceptual LPIPS gain is the thesis-relevant effect** — it's exactly the fine-detail recovery AbsGS is designed for.

---

## Setup

| | |
|---|---|
| **Method (test arm)** | gsplat engine + `use_principal_point=true` + **AbsGrad** (`absgrad=true`, `densify_grad_threshold=0.0008`) |
| **Control** | `test_gsplat_full` — gsplat engine + `use_principal_point=true` + **default** (signed-gradient) densification |
| **Plots** | FIP plot_461 … plot_467 (7 plots) |
| **Iterations** | 30k (the 15k checkpoint is evaluated from the same run) |
| **Eval split** | FIP camera-index split (cam_11 / cam_12 held out as test) |
| **Experiment names** | test arm `test_absgrad_v2` (all 7 plots complete; 463/467 were redone via the seg-fix job and consolidated back into `test_absgrad_v2`); control `test_gsplat_full` |
| **Scripts** | `scripts/fip_test_absgrad_split_a_job.sh` (461–463), `…_split_b_job.sh` (464–467), seg-fix `scripts/fip_absgrad_seg_fix_463_467_job.sh` |

The control is the strongest non-AbsGrad baseline: gsplat already matched the old `diff-gaussian-rasterization` engine on quality (it only added speed), so "gsplat-default densification" isolates the densification change as the *only* difference vs the AbsGrad arm.

---

## 1. Reconstruction quality @ 30k — three engines/methods

Three arms, all at resolution 1 with `use_principal_point=true`:
- **AbsG** = gsplat + AbsGrad densification (the test arm)
- **gs-def** = gsplat + default densification (`test_gsplat_full`, primary control)
- **dgs-def** = `diff-gaussian-rasterization` + default densification (`test_diffgs_full`, the original engine)

| plot | PSNR AbsG | PSNR gs-def | PSNR dgs-def | SSIM AbsG | SSIM gs-def | SSIM dgs-def | LPIPS AbsG | LPIPS gs-def | LPIPS dgs-def |
|------|----------:|------------:|-------------:|----------:|------------:|-------------:|-----------:|-------------:|--------------:|
| 461 | 26.29 | 25.76 | 25.79 | 0.846 | 0.829 | 0.830 | 0.188 | 0.216 | 0.213 |
| 462 | 30.12 | 29.54 | 29.51 | 0.912 | 0.902 | 0.902 | 0.163 | 0.188 | 0.188 |
| 463 | 27.96 | 27.50 | 27.45 | 0.894 | 0.882 | 0.882 | 0.173 | 0.196 | 0.195 |
| 464 | 27.62 | 27.22 | 27.19 | 0.887 | 0.874 | 0.874 | 0.187 | 0.222 | 0.222 |
| 465 | 28.61 | 28.21 | 28.17 | 0.890 | 0.881 | 0.880 | 0.181 | 0.204 | 0.203 |
| 466 | 29.83 | 29.16 | 29.20 | 0.902 | 0.891 | 0.891 | 0.160 | 0.184 | 0.183 |
| 467 | 30.50 | 29.84 | 29.85 | 0.923 | 0.912 | 0.912 | 0.152 | 0.179 | 0.179 |
| **mean** | **28.70** | **28.17** | **28.17** | **0.893** | **0.881** | **0.881** | **0.172** | **0.198** | **0.198** |

**Two readings:**
1. **gs-def ≈ dgs-def on every metric** (means identical to 2 d.p.: 28.17 / 0.881 / 0.198). The render *engine* does not change quality — gsplat only added ~1.77× training speed. This confirms the control is fair: the AbsGrad comparison isolates **densification**, not engine.
2. **AbsGrad beats both baselines on all 7 plots, all 3 metrics** — vs the gs-def control: mean **PSNR +0.53 dB**, **SSIM +0.012**, **LPIPS −0.026** (per-plot ΔPSNR +0.40 … +0.67, never negative). LPIPS improves a steady ~0.026, the largest *relative* effect and the one that captures perceptual fine-detail.

---

## 2. 15k vs 30k (within AbsGrad)

Because the LR and densification schedules are anchored to fixed step counts, iteration-15000 of a 30k run is identical to a standalone 15k run.

| metric | 15k mean | 30k mean | Δ (30k − 15k) |
|--------|---------:|---------:|--------------:|
| PSNR | 28.72 | 28.70 | **−0.02** (flat) |
| LPIPS | 0.175 | 0.172 | −0.004 |

PSNR/SSIM are statistically identical at 15k and 30k (on several plots 15k is marginally *higher* — pure noise); LPIPS is only ~0.004 better at 30k. **The extra 15k iterations (≈ doubling train time) buy essentially nothing**, so **15k is the efficient operating point.**

---

## 3. Segmentation quality (eval_2d) — AbsGrad vs gsplat-default

2D segmentation metrics vs the manually-labelled GT mask (one GT camera per plot). All 7 plots.

| plot | IoU abs | IoU gs | ΔIoU | F1 abs | F1 gs | ΔF1 |
|------|--------:|-------:|-----:|-------:|------:|----:|
| 461 | 0.565 | 0.518 | +0.048 | 0.722 | 0.682 | +0.040 |
| 462 | 0.718 | 0.695 | +0.023 | 0.836 | 0.820 | +0.016 |
| 463 | 0.590 | 0.509 | +0.081 | 0.742 | 0.674 | +0.068 |
| 464 | 0.695 | 0.685 | +0.010 | 0.820 | 0.813 | +0.007 |
| 465 | 0.693 | 0.693 | +0.000 | 0.819 | 0.819 | +0.000 |
| 466 | 0.304 | 0.298 | +0.006 | 0.467 | 0.460 | +0.007 |
| 467 | 0.706 | 0.717 | −0.011 | 0.827 | 0.835 | −0.008 |
| **mean** | **0.610** | **0.588** | **+0.022** | **0.748** | **0.729** | **+0.019** |

**AbsGrad wins or ties on 6 of 7 plots.** The per-plot precision/recall split shows the mechanism: AbsGrad lifts **recall** (it finds more heads — e.g. 461 recall 0.650 vs 0.597) at a small precision cost, so net IoU/F1 go up. **463 is the biggest single win** (+0.081 IoU); 461 is also a clear win; 464/465 are near-ties. **467 is the lone slight loss** (−0.011 IoU): there AbsGrad's extra recall tipped into over-counting (pred 328 heads vs 216 GT, count-error +0.52), so precision dropped enough to net-lose — consistent with the single-GT-camera caveat below.

> Note: 466 has low absolute IoU (~0.30) for both methods — a hard plot — but AbsGrad is still ahead. This is a plot-difficulty issue, not a method issue.

---

## Why AbsGrad is better (intuition)

Default 3DGS densification decides where to add Gaussians from the **signed** screen-space gradient. On fine, high-frequency structure (wheat awns, head edges), opposing per-pixel sub-gradients **cancel** in the signed sum ("gradient collision"), so those regions never get densified → they stay blurry. AbsGS accumulates the **absolute** gradient magnitude for the split/clone *decision* (the optimisation/movement gradient keeps its sign), so fine detail gets the Gaussians it needs. Result: sharper renders (LPIPS↓) and crisper instance masks (IoU/recall↑), with the Gaussian count staying in the normal range.

---

## Caveats

- **Seg is a single GT camera per plot** — treat small seg deltas (464, 465, and the 467 loss) as ties/noise; 461 and 463 are the robust wins.
- **463 & 467 seg** — their original `test_absgrad_v2` flashsplat seg was cut by a 6 h per-plot timeout, so they were redone via the seg-fix job and consolidated back into `test_absgrad_v2`. Their recon is from that retrain (≤0.01 dB vs the original). Tables now 7/7.
- **360 videos not produced** — Euler ffmpeg is missing `libopenh264.so.5`; frames render fine, only the mp4 stitch fails (cosmetic, no effect on any metric).
- Numbers are from the `test_absgrad_v2` / `test_gsplat_full` runs at resolution 1; principal-point fix on (`use_principal_point=true`) for both arms, so the comparison isolates densification only.

---

## Status / reproducibility

- **Done:** recon (7/7) + seg (7/7). AbsGrad validated as a strong improvement over gsplat-default on FIP (recon 7/7, seg 6/7 wins/ties, one near-tie loss on 467).
- **Config knob:** when `absgrad=true`, `densify_grad_threshold` must be raised ~3–4× (0.0008 here vs the 0.0002 default) or it over-densifies — see `docs/DENSIFICATION_OPTIONS.md` §8.
- **Phone:** not yet run — AbsGrad on phone sessions is the next step.
