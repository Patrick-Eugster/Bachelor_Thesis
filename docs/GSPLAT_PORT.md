# gsplat Port (branch: `gsplat-switch`)

Status: **WIP — implemented + smoke-tested locally, NOT yet full-30k validated.**
Rollback any time with `git checkout round4-signfix-good` (the Round-4 known-good tag).

## What this is

Swaps the rendering engine from the custom `diff-gaussian-rasterization` CUDA submodule to
[**gsplat**](https://github.com/nerfstudio-project/gsplat) (the Nerfstudio/splatfacto backend).

**Why bother, given Round 4 already hit 28.17 PSNR?** The Round-4 sign-fix corrected the *projection
matrix* (cx/cy now honored via an asymmetric frustum), but the diff-gaussian rasterizer still does its
tile-binning / frustum-culling with scalar `tanfovx/tanfovy`, which assume a *symmetric* frustum — a
residual sub-pixel approximation at the image borders. gsplat's kernel takes the full intrinsic matrix
`K` (with cx/cy) natively, so the whole projection→rasterization path becomes self-consistent. Bonus:
gsplat is actively maintained and Blackwell-ready. See [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md) for the
projection background.

> Note: empirically the Round-4 FFT shift was already **(0,0)** on every plot, so the residual
> approximation has **no measured headroom** — the motivation for gsplat is code cleanliness +
> future-proofing, not a quality gap. It is a deliberate, optional cleanup, not a thesis blocker.

## What changed (3 code edits + 1 env fix)

### 1. `src/gaussians/gaussian_renderer/__init__.py` — new `render()`
- `render()` now calls `gsplat.rasterization(...)` instead of `GaussianRasterizer`.
- The **old diff-gaussian path is preserved verbatim** as `render_diffgs()` right below it, so we can
  A/B or roll back by swapping one import. Nothing was deleted.
- `flashsplat_render()` (the segmentation engine) is **untouched** — still flashsplat.

### 2. `src/gaussians/scene/gaussian_model.py` — `add_densification_stats`
- Now squeezes the camera dim, because gsplat returns `means2d` as `[C, N, 2]` (C=1) whereas the old
  path returned `[N, 3]`. One `if grad.dim() == 3: grad = grad.squeeze(0)`.

### 3. `~/.bashrc` — torch import fix (container-only, not committed)
- Local `import torch` was crashing: the container's `/etc/ld.so.conf.d/hpcx.conf` puts HPC-X's
  `libucc.so.1` on the path, which needs the symbol `ucs_config_doc_nop` that only exists in HPC-X's
  `libucs.so.0` — but the *system* `libucs.so.0` resolves first. Fix: `export
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH` so the matching libucs wins. HPC-X is only
  actually used on Euler, never locally.

## Three correctness details (where a naive port silently breaks)

| Detail | Handling |
|---|---|
| **viewmat format** | 3DGS stores `world_view_transform` pre-transposed for glm; gsplat wants the plain world→cam, so we `.transpose(0,1)` to undo it. |
| **intrinsics K** | `fx = W/(2·tan(FoVx/2))`, `fy = H/(2·tan(FoVy/2))`; `cx,cy` from the Camera when `use_principal_point` set them, else image center. This is what makes the pixel-shift fix **native** — no asymmetric-frustum projmatrix needed. |
| **densification grad scale** | gsplat's `means2d` grad is in **pixels**; INRIA's `screenspace_points` grad is in **NDC**, and `densify_grad_threshold=0.0002` is calibrated for NDC. A backward hook rescales gsplat's grad by `(W/2, H/2)` so densification stays identical to Round 4. |

Plus two guards found during testing:
- **`MiniCam` has no `cx`/`cy`** (used by render_360 / viewer) → `getattr(..., "cx", None)` falls back
  to image center (their original symmetric behavior). Without this, those paths `AttributeError`.
- **`retain_grad()` under `torch.no_grad()`** (render/metrics/eval) errors because `means2d` has
  `requires_grad=False` → guarded with `if means2d.requires_grad:`.

## Coverage — what runs on gsplat vs flashsplat

Because `render()` is shared, the port reaches more than just train+render:

| Pipeline path | Function called | Engine now |
|---|---|---|
| Training | `render()` | ✅ gsplat |
| Render step | `render()` | ✅ gsplat |
| Eval step 6 (RGB overlay) | `render()` | ✅ gsplat |
| render_360 (plain RGB) | `render()` | ✅ gsplat |
| Viewer (viser) | `render()` | ✅ gsplat |
| **Segmentation_3d (step 4)** | `flashsplat_render()` | ❌ **still flashsplat (unchanged)** |
| render_360 (per-head seg coloring) | `flashsplat_render()` | ❌ still flashsplat |

**Segmentation is deliberately not ported** — flashsplat is its own independent CUDA submodule. Decide
separately whether to leave it (carries the same residual sub-pixel approximation as diff-gaussian did)
or port/patch it later.

## Smoke test (plot_461, 1000 iters, gsplat + `use_principal_point=true`)

| Metric | gsplat @ 1000 iter |
|---|---:|
| PSNR | 22.60 |
| SSIM | 0.699 |
| LPIPS | 0.427 |

Sanity-only (1000 iters, not 30k → not comparable to Round-4's 25.78 on plot_461). Confirms: trains,
loss drops, **densification fires** (31,949 → 61,987 points = grad-rescale path works), render produces
correctly-aligned images (a broken/shifted render would land ~12–15 dB). gsplat first-use JIT-compiled
its sm_120 kernels in ~7 min (one-time).

**Render-speed observation:** during training, render was ~9 ms/iter; during the inference render step,
the first image took ~340 s and the rest ~5 s each. Almost certainly gsplat re-autotuning per unique
image size — the FIP plot mixes sub-session resolutions (`_1_`, `_6_`, base). One-time per size, then
cached; not a per-frame cost. Worth confirming on Euler.

## Next steps

1. **Full 30k run on plot_461** → compare PSNR/SSIM/LPIPS + FFT shift vs Round-4 (plot_461 = 25.78).
   gsplat should match or slightly beat it. Run on Euler (avoids local 16 GB VRAM limit).
2. If clean: full 7-plot rerun as `experiment_name=paper_bench_30k_gsplat`.
3. Decide flashsplat fate (leave / port / patch).
4. Verify render_360 + viewer still render correctly (only train/render were smoke-tested).

## Rollback

```bash
git checkout round4-signfix-good   # back to Round-4 diff-gaussian known-good
git checkout main                  # abandon the gsplat experiment entirely
```
The diff-gaussian engine also stays callable in-tree as `render_diffgs()` without any checkout.
