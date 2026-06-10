# Densification options for wheat 3DGS — Default vs AbsGrad vs MCMC

> **TL;DR** — Vanilla 3DGS (what we run today) decides "where to add more Gaussians" with a
> gradient rule that **cancels out on fine, detailed regions** (the "gradient collision" bug) —
> so small wheat-head detail and thin awns come out blurry. Two better densification strategies
> exist in **gsplat** (which we already use for rendering on the `gsplat-switch` branch):
> **AbsGrad** (fixes fine detail, ~free) and **MCMC** (fixes speed/VRAM, fixed Gaussian budget).
> They are **alternatives, not a stack** — pick one and A/B it. For our fine-detail-first goal,
> AbsGrad is the most on-target single change; MCMC is the speed/memory play.

This doc explains, for someone new to 3DGS internals:
1. what "densification" is and exactly what our current code does,
2. what **AbsGrad** changes and why it recovers fine detail,
3. what **MCMC** changes and why it is faster / lighter / more robust on sparse views,
4. how each differs from the current approach and what switching would touch,
5. which to use for the wheat use case.

Everything here keeps the output a **standard Gaussian point cloud**, so the downstream
FlashSplat 3D segmentation (`src/segmentation_3d/run_3d_seg.py`) is unaffected by any of these
choices. Methods that change the primitive (2DGS, Scaffold/anchor-GS, neural-decoded Gaussians)
are deliberately *not* covered here because they would break segmentation.

---

## 0. What "densification" even is

3DGS starts from the sparse SfM point cloud (a few thousand points) and must grow it into
millions of Gaussians that reconstruct the scene. **Densification** is the rule that, during
training, decides **which Gaussians to add (and remove)** so detail appears where it's needed
and memory isn't wasted where it isn't. It is the single biggest lever on **final quality**,
**training time**, and **VRAM**. The optimizer tunes each Gaussian's position/color/shape;
densification decides *how many* Gaussians exist and *where*.

---

## 1. What we do today — the original 3DGS "Default" strategy

Call sites in our code:
- Training loop: [`src/reconstruction/vanilla_3dgs/train_vanilla_3dgs.py`](../src/reconstruction/vanilla_3dgs/train_vanilla_3dgs.py) lines ~118-129
- Logic: [`src/gaussians/scene/gaussian_model.py`](../src/gaussians/scene/gaussian_model.py) — `add_densification_stats`, `densify_and_prune`, `densify_and_clone`, `densify_and_split`, `reset_opacity`

What happens each iteration (while `iteration < densify_until_iter`, default 11000):

1. **Accumulate a gradient signal** (`add_densification_stats`):
   for every Gaussian, add the **norm of its screen-space positional gradient**
   (`‖∇‖` of where its 2D center wants to move), and count how many views it appeared in.
   The per-Gaussian criterion is the **average** of this over time.

2. Every `densification_interval` iterations (`densify_and_prune`), compute
   `grad = accumulated / count` and act on it:
   - **Clone** (`densify_and_clone`): Gaussian has **high grad AND is small** → duplicate it.
     (Under-reconstructed: a small region needs more coverage.)
   - **Split** (`densify_and_split`): Gaussian has **high grad AND is large** → replace with 2
     smaller ones. (Over-reconstructed: one big blob is straddling detail.)
   - **Prune**: remove Gaussians that are nearly transparent (`opacity < threshold`) or too big
     on screen / in world.

3. Periodically **reset all opacities** low (`reset_opacity`) to flush floaters; the optimizer
   then re-grows the ones that are truly needed.

Key properties:
- The "do I need more detail here?" decision is **one scalar**: the averaged, *direction-aware*
  screen-space gradient vs `densify_grad_threshold` (0.0002).
- Gaussian count **grows without an upper bound** (until `densify_until_iter`) — this is why VRAM
  peaks unpredictably around iter 9000-11000 (see CLAUDE.md VRAM notes).

### The flaw that hurts wheat: "gradient collision"

In step 1 the sub-gradients from individual pixels are summed **with their sign/direction**.
On a **detailed, high-frequency region** (a wheat head with texture, or a thin awn), one big
Gaussian is pulled **left by some pixels and right by others** at the same time. Those opposing
sub-gradients **cancel** to nearly zero. The averaged criterion then reads "low gradient → this
Gaussian is fine" and **never splits it** — even though it's blurring real detail underneath.
Result: detail-rich areas stay **over-reconstructed (blurry)**. This is precisely the wheat
problem (small heads + fine awns).

---

## 2. AbsGrad (AbsGS) — the fine-detail fix

**Paper:** *AbsGS: Recovering Fine Details for 3D Gaussian Splatting* (ACM MM 2024).
**One-line idea:** accumulate the gradient **magnitude regardless of direction** — i.e. sum the
**absolute values** of the per-pixel sub-gradients (a "homodirectional" gradient) instead of the
signed vector sum.

Why it works: the absolute operation removes the cancellation. A big Gaussian straddling a
detailed region now shows a **high** criterion (because lots of pixels are pulling it hard, even
if in opposing directions) → it gets **split** → fine detail is recovered. It does **not** add
Gaussians where they aren't needed, so it doesn't blow up the count.

How it differs from current:
- **Same machinery** (clone/split/prune/reset, gradient threshold, unbounded count).
- **Only the gradient criterion changes** from "norm of the signed sum" to "sum of absolute
  values". One number, computed differently.

Cost / risk:
- **Compute cost ≈ zero** (it's a different reduction of gradients we already have).
- The effective threshold scale changes slightly, so `densify_grad_threshold` usually needs
  re-tuning (the AbsGS authors use a larger threshold, ~0.0004-0.0008, because absolute grads
  are larger). This is the one knob to sweep.

In gsplat this is the flag **`absgrad=True`** on `gsplat.rasterization(...)` plus reading
`meta["means2d"].absgrad` (instead of `.grad`) as the densification signal. gsplat's
`DefaultStrategy` exposes it directly.

**Verdict for wheat:** the most **on-target** single change for "best detail" — it is literally
the method built to recover what vanilla blurs. Top pick for the fine-detail requirement.

---

## 3. MCMC — the speed / memory / robustness fix

**Paper:** *3D Gaussian Splatting as Markov Chain Monte Carlo* (NeurIPS 2024); shipped as
`MCMCStrategy` in gsplat.
**One-line idea:** stop using clone/split/prune/reset heuristics. Instead treat the Gaussians as
**samples of a probability distribution** and refine them under a **fixed budget**.

What it does on a schedule:
- **Relocate "dead" Gaussians**: any Gaussian whose opacity has decayed to ~0 is wasted; teleport
  it to a high-density / high-opacity area that needs more samples, in a way that keeps the
  rendered image unchanged at the moment of the move.
- **Inject noise** into positions so Gaussians explore and escape bad local minima (this is the
  "Monte Carlo" part — it samples rather than greedily splits).
- **Fixed budget `cap_max`**: you choose the maximum number of Gaussians up front. The strategy
  fills up to that budget and then only relocates — it never grows past it.

How it differs from current:
- **Replaces the entire densification mechanism.** No gradient threshold, no clone, no split, no
  opacity reset. (So `absgrad` is irrelevant under MCMC — MCMC doesn't look at the gradient
  criterion at all.)
- **Count is bounded and chosen by you** → VRAM is predictable (no more "peaks at 15-16 GB,
  hangs at iter 9000"). Reported ~40% faster training and ~5× less memory than the original.
- Empirically **more robust on sparse / limited-view captures** (our ~93 phone/FIP images),
  because it won't over-spawn Gaussians in weakly-observed regions.

Cost / risk:
- The **`cap_max` budget is the critical knob**: too low and you starve fine detail (the opposite
  of what we want); set it generously (≈ the Gaussian count a Default run lands at, or higher) so
  detail isn't capped.
- Different hyperparameters than Default (`noise_lr`, `refine_every`, `min_opacity`,
  `refine_start/stop_iter`) — a small sweep is needed to match/beat Default quality.

**Verdict for wheat:** the **speed + VRAM** play, and often a quality win on sparse views — but its
fixed budget means you must size `cap_max` so it doesn't limit fine detail. Best when training
time / 16 GB VRAM is the binding constraint.

---

## 4. Side-by-side

| | **Default (current)** | **Default + AbsGrad** | **MCMC** |
|---|---|---|---|
| Decides density by | signed screen-space gradient | **absolute** screen-space gradient | opacity-based relocation |
| Fixes "gradient collision" blur? | ❌ no | ✅ **yes** | n/a (different mechanism) |
| Gaussian count | unbounded (grows) | unbounded (grows) | **fixed budget `cap_max`** |
| Training speed vs current | baseline | ~baseline | **~40% faster** |
| VRAM | unpredictable peak | unpredictable peak | **bounded / predictable** |
| Best at | — | **fine detail** | **speed / VRAM / sparse views** |
| Main knob to tune | `densify_grad_threshold` | `densify_grad_threshold` (larger) | `cap_max`, `noise_lr` |
| Segmentation-compatible | ✅ | ✅ | ✅ |
| In gsplat | `DefaultStrategy()` | `DefaultStrategy(absgrad=True)` | `MCMCStrategy(cap_max=...)` |

**They are alternatives, not a stack.** Pick one per run and A/B against the diffgs/gsplat
baselines (`test_diffgs_full` / `test_gsplat_full`).

---

## 5. Recommendation for the wheat thesis

1. **Primary (fine detail):** `DefaultStrategy(absgrad=True)` — directly targets the blurry-detail
   failure mode on wheat heads/awns, ~free, keeps everything else identical. Sweep
   `densify_grad_threshold` (try ~0.0004-0.0008).
2. **Speed/VRAM alternative:** `MCMCStrategy` with a generous `cap_max` — if the binding
   constraint becomes the ~2h training time or the 16 GB ceiling. Size `cap_max` so detail isn't
   starved.
3. **A/B both** against the current baselines on the same plots; compare `results.json`
   (PSNR/SSIM/LPIPS) **and** the `eval_2d/metrics_2d.json` segmentation metrics (detail that
   doesn't survive into correct head segmentation isn't worth much).

If maximum detail is the thesis headline regardless of speed, also consider **Mip-Splatting**
(anti-aliasing for thin structures at varying zoom) on top of AbsGrad — but that is a separate
change, not a gsplat flag, and is out of scope for this doc.

---

## 6. What switching would touch in our code (high level)

We already render on gsplat (`gsplat-switch` branch, see [`GSPLAT_PORT.md`](GSPLAT_PORT.md)), but
**training still densifies with our hand-rolled INRIA logic** in
[`gaussian_model.py`](../src/gaussians/scene/gaussian_model.py) (`add_densification_stats` +
`densify_and_prune`). Adopting either strategy means moving the densification decision into
gsplat's `Strategy` API inside
[`train_vanilla_3dgs.py`](../src/reconstruction/vanilla_3dgs/train_vanilla_3dgs.py):

- **AbsGrad:** pass `absgrad=True` to the `rasterization()` call in
  [`gaussian_renderer/__init__.py`](../src/gaussians/gaussian_renderer/__init__.py) `render()`,
  and use `meta["means2d"].absgrad` as the densification signal (either via gsplat
  `DefaultStrategy` or by changing what `add_densification_stats` accumulates). Smallest change.
- **MCMC:** replace the clone/split/prune/reset block in the training loop with gsplat's
  `MCMCStrategy.step_pre_backward` / `step_post_backward` callbacks and an Adam over the strategy's
  parameter dict; remove `reset_opacity` and the size/opacity prune (MCMC handles all of it).
  Larger change to the training loop, but contained — it does not touch render, eval, or
  segmentation.

Both are opt-in and should be benchmarked before becoming default, exactly like the
`use_principal_point` and gsplat-render changes were.

---

## 7. Update (2026-06-09): how the gsplat benchmark changes the priority

The gsplat engine benchmark ([`GSPLAT_VS_DIFFGS_RESULTS.md`](GSPLAT_VS_DIFFGS_RESULTS.md)) showed
gsplat already trains **~1.77× faster** (~89 → ~50 min/plot) at **identical quality** (PSNR 28.17,
seg IoU 0.588). The render engine and the densification strategy are **different layers** — gsplat
renders the *same* Gaussians faster; it does not change *what* is reconstructed. So:

- **The speed motivation for MCMC is largely satisfied by gsplat.** MCMC's remaining unique value is
  **bounded VRAM** (only matters if the 16 GB ceiling still bites) and **sparse-view robustness**
  (most relevant for **phone**, not FIP). → **MCMC demoted to "optional, mainly for phone."**
- **AbsGrad stays the priority lever.** gsplat did **not** improve quality (PSNR unchanged), and the
  thesis goal is **fine detail** (awns, small heads) with clear headroom (seg IoU only ~0.59).
  AbsGrad targets exactly that, nearly for free. → **Reframe the planned A/B/C as
  gsplat-Default vs gsplat-AbsGrad** as the headline arm; add gsplat-MCMC only if VRAM/phone needs it.

One-line: **gsplat answered "faster"; AbsGrad answers "sharper" — and the thesis cares about sharper.**

---

## 8. AbsGrad IMPLEMENTED (2026-06-10) — opt-in flag, off by default

AbsGrad is now wired in as a toggle. **Default is off → no-flag runs are byte-identical to the
current vanilla path** (verified: with the flag off, gsplat never populates `means2d.absgrad` and
densification reads the same signed `.grad` as before). We kept the existing INRIA clone/split/prune
machinery and only swapped *which gradient feeds the split decision* — no gsplat `Strategy` refactor
was needed (that's only required for MCMC).

**What changed (4 spots):**
1. [`gaussian_renderer/__init__.py`](../src/gaussians/gaussian_renderer/__init__.py) `render()` —
   new `absgrad=False` arg, passed through to `gsplat.rasterization(absgrad=...)`. Only the train
   loop passes `True`; render/eval/viewer leave it `False`.
2. [`gaussian_model.py`](../src/gaussians/scene/gaussian_model.py) `add_densification_stats(..., use_absgrad=False)` —
   reads `means2d.absgrad` instead of `.grad` when on.
3. [`train_vanilla_3dgs.py`](../src/reconstruction/vanilla_3dgs/train_vanilla_3dgs.py) — renders with
   `absgrad=opt.absgrad`, and rescales `.absgrad` from pixels → NDC (camera `W/2, H/2`) before
   accumulating, because `.absgrad` skips the pixel→NDC autograd hook that `.grad` gets in `render()`.
   This keeps `densify_grad_threshold` on the same scale as the default path.
4. Config: `absgrad` flag in `OptimizationParams` (auto `--absgrad` CLI), `reconstruction.absgrad`
   in [`vanilla_3dgs.yaml`](../configs/reconstruction_seg3d/reconstruction/vanilla_3dgs.yaml), and
   `absgrad_flag` plumbed through `run_reconstruction.py`'s Train step.

**How to run the A/B (gsplat-Default vs gsplat-AbsGrad):**
```bash
# A — current vanilla (control)
python src/run_reconstruction.py plot=plot_461 run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=30000 reconstruction.use_principal_point=true experiment_name=absA_default

# B — AbsGrad on. IMPORTANT: raise the threshold (~3-4x) — absgrad magnitudes are larger,
#     so the default 0.0002 would massively over-densify and OOM.
python src/run_reconstruction.py plot=plot_461 run_train=true run_render=true run_metrics=true \
  reconstruction.iterations=30000 reconstruction.use_principal_point=true \
  reconstruction.absgrad=true reconstruction.densify_grad_threshold=0.0008 experiment_name=absB_absgrad
```
Then compare PSNR/SSIM/LPIPS + Gaussian count + seg `eval_2d/metrics_2d.json`. **The threshold is the
one knob to tune** — start at 0.0008 and adjust so the final Gaussian count lands near the default
run's (too low → OOM/over-dense; too high → no benefit). AbsGrad works on the gsplat engine only
(the `WHEAT_RENDERER=diffgs` fallback ignores it). FlashSplat segmentation is unaffected — still a
standard Gaussian cloud.

---

## 9. Other 3DGS variants we evaluated (and why they don't fit)

Beyond densification strategy, several well-known 3DGS variants were considered as alternatives to
vanilla. Two hard filters decide fit for **this** project:

- **Filter A — segmentation-compatible?** FlashSplat assigns a label to each Gaussian in a *fixed,
  explicit* cloud. Anything that **decodes Gaussians from a neural network** (no stable per-Gaussian
  identity) breaks 3D segmentation.
- **Filter B — does it target OUR scene?** Our captures are **fixed-distance, high-texture, static,
  fine-detail** (wheat awns/edges). Methods built for textureless, dynamic, or free-viewpoint
  multi-scale scenes solve problems we don't have.

| Method | What it targets / does | Verdict for wheat3dgs |
|---|---|---|
| **Mip-Splatting** (Yu et al. 2023) | Scale/zoom **aliasing**: a 3D low-pass (smoothing) filter + a 2D Mip filter replacing 3DGS's dilation, so quality is stable across resolutions/distances. | ✅ **Segmentation-safe**, and partly **free**: gsplat exposes it via `rasterize_mode="antialiased"`. But our cameras sit at ~fixed distance, so multi-scale aliasing is a minor issue → **low expected gain, but cheap to A/B** (could become a `mip`/`antialiased` opt-in flag like `absgrad`). |
| **GaussianPro** (Cheng et al. 2024) | Progressive propagation: uses rendered depth/normal + patch-match (MVS-style) priors and planar constraints to grow Gaussians in **under-reconstructed / textureless** regions. | ⚠️ **Off-target** — wheat is *high-texture* (edges everywhere), the opposite of its use case; also meaningful integration effort. Segmentation-safe but little to gain → **skip**. |
| **Scaffold-GS** (Lu et al. 2024) | Anchor points hold features; an **MLP decodes** the attributes of k view-dependent neural Gaussians per anchor, spawned each frame. Compact + high quality. | ❌ **Breaks segmentation** (Filter A): no fixed explicit per-Gaussian set with stable identity for FlashSplat to label. Rule out unless segmentation is re-architected. |
| **Deformable 3DGS** (Yang et al. 2024) / 4DGS | A **deformation MLP** maps canonical Gaussians + time → deformed positions, for **dynamic/temporal** scenes. | ❌ **Not our problem** (Filter B): reconstruction is *static* per capture session; the deformation field also complicates the cloud. Only relevant if we later model *growth over time* — a different project. |
| **LV-3DGS** | ❓ Not a name I can confidently place — it overlaps with several niche pruning / level-of-detail variants (e.g. LightGaussian, LP-3DGS). | ❓ **Needs clarification** (which paper/link). Not assessed here rather than guess. If it's a *pruning/LoD* method it's likely segmentation-safe but aimed at memory/speed, which gsplat already largely solved. |

**Takeaway:** of these five, only **Mip-Splatting** is both segmentation-safe and on-target, and even
it is a minor, cheap-to-try gain — not a fine-detail lever. **AbsGrad remains the best fit** because it
directly attacks the awn/edge detail loss while keeping the standard explicit Gaussian cloud.
Scaffold-GS and Deformable are ruled out by our segmentation / static-scene constraints; GaussianPro
targets a low-texture problem we don't have. This is the same reasoning that ruled out 2DGS and other
neural-decoded representations earlier in this doc.
