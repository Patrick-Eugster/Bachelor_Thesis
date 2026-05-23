# Pixel-Shift Bug in train / render / metrics Pipeline

Investigation into the supervisor-reported pixel-shifting issue that makes our 3DGS metrics worse than the original Wheat3DGS paper's (which used splatfacto). This document records the findings from a careful read of the train, render, and metrics code paths.

## Summary

The vanilla 3DGS pipeline in this repo **silently drops the principal point (cx, cy) from COLMAP/Agisoft camera intrinsics** and forces it to image center. For our FIP plots, the actual principal point is offset by tens of pixels (up to ~90 px) from center, so every rendered image is misaligned with its ground-truth image by that same offset. PSNR/SSIM/LPIPS are computed on misaligned pairs, which destroys the numbers even when the reconstruction itself is fine.

Splatfacto (Nerfstudio's 3DGS, used by the paper) does not have this issue because it uses **gsplat**, which accepts the full intrinsic matrix including cx/cy.

## The bug: principal point (cx, cy) is silently dropped

**Where:** [src/gaussians/scene/dataset_readers.py:89-99](../src/gaussians/scene/dataset_readers.py#L89-L99)

```python
if intr.model == "SIMPLE_PINHOLE":
    focal_length_x = intr.params[0]
    FovY = focal2fov(focal_length_x, height)
    FovX = focal2fov(focal_length_x, width)
elif intr.model == "PINHOLE":
    focal_length_x = intr.params[0]
    focal_length_y = intr.params[1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
```

COLMAP intrinsics actually contain **4 numbers** (`fx, fy, cx, cy` for PINHOLE; `f, cx, cy` for SIMPLE_PINHOLE). This code only reads the focal-length params, **never reads `cx`/`cy`**. The principal point is thrown away.

The full chain that depends on it:
- `CameraInfo` only carries `FovX, FovY` — no cx/cy field exists ([dataset_readers.py:29-41](../src/gaussians/scene/dataset_readers.py#L29-L41))
- `Camera` only stores `FoVx, FoVy` and computes the projection matrix from those ([cameras.py:57](../src/gaussians/scene/cameras.py#L57))
- `getProjectionMatrix` builds a **symmetric frustum** (`left=-right`, `bottom=-top`), forcing principal point to image center: [graphics_utils.py:51-71](../src/gaussians/utils/graphics_utils.py#L51-L71)
- The CUDA rasterizer takes only `tanfovx, tanfovy, projmatrix` — no cx/cy slot ([gaussian_renderer/__init__.py:37-53](../src/gaussians/gaussian_renderer/__init__.py#L37-L53))

So every rendered pixel is projected as if the camera looked straight through `(W/2, H/2)`, but the GT image was actually captured with a sensor whose optical axis hits a different pixel. **GT vs render are shifted by exactly `(cx − W/2, cy − H/2)` pixels.**

## Evidence from our data

Measured directly from each plot's `sparse/0/cameras.txt`:

| Plot | W×H | cx | cy | Δcx (px) | Δcy (px) |
|---|---|---|---|---|---|
| plot_461 | 4095×2996 | 1956 | 1480 | **−91.5** | −18.0 |
| plot_462 | 4095×2995 | 1969 | 1513 | **−78.5** | +15.5 |
| plot_463 | 4094×2998 | 1987 | 1499 | −60.0 | +0.0 |
| plot_464 | 4093×2997 | 1996 | 1438 | −50.5 | **−60.5** |
| plot_465 | 4095×2998 | 2027 | 1524 | −20.5 | +25.0 |
| plot_466 | 4096×2996 | 2001 | 1506 | −47.0 | +8.0 |
| plot_467 | 4094×2996 | 1991 | 1483 | −56.0 | −15.0 |

Every FIP plot is off by **tens to ~90 pixels** in x, up to 60 px in y. At 4K width that's ~1–2% per axis — easily enough to destroy PSNR/SSIM on a sharp scene like wheat heads, and exactly the kind of "renders look right but metrics tank" symptom the supervisor described.

## Why phone metrics didn't show the same bug (much)

For the phone data, COLMAP's `image_undistorter` (and Agisoft's equivalent) **recenter the principal point** during undistortion:

```
phone field_A/20250609 colmap : W=4032 H=3024 cx=2016 cy=1512  → EXACTLY centered
phone field_A/20250609 agisoft: W=3964 H=2926 cx=1982 cy=1463  → EXACTLY centered
phone field_D/20250523 agisoft: W=3846 H=2924 cx=1923 cy=1462  → EXACTLY centered
```

So for the phone benchmark this specific bug didn't fire — both sides (COLMAP + Agisoft) deliver `cx=W/2, cy=H/2`. The phone PSNR being low (14–16) must come from somewhere else (scene difficulty, llffhold split, undertraining), but that's a different problem.

## Why splatfacto doesn't have this bug

Splatfacto (Nerfstudio) runs on **gsplat**, which takes the full intrinsic matrix `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]` as input and bakes cx/cy into projection. Vanilla `diff-gaussian-rasterization` (what this repo uses) only takes `tanfovx, tanfovy, projmatrix`, and the CUDA code assumes principal point at center. So when the supervisor says "splatfacto doesn't have the issue" — yes, that's literally because gsplat plumbs cx/cy and diff-gaussian-rasterization doesn't.

## Other things checked and ruled out

- **PIL resize**: `PILtoTorch` uses `Image.resize(resolution)` without a resample arg ([general_utils.py:21-27](../src/gaussians/utils/general_utils.py#L21-L27)). Default is BICUBIC on modern Pillow — doesn't introduce a shift, only mild aliasing.
- **Resolution scaling at resolution=1**: focal length scales implicitly with the FoV formula since both W and f scale together, so no shift from rescaling.
- **render.py and metrics.py**: render.py just saves render + GT side by side ([render.py:31-35](../src/reconstruction/render.py#L31-L35)) and metrics.py loads both back at the same resolution ([metrics.py:24-34](../src/reconstruction/metrics.py#L24-L34)). No re-resizing, no flip, no per-channel offset — these are clean.
- **Eval split**: the `cam_idx > 10` test split for FIP cameras ([dataset_readers.py:191-197](../src/gaussians/scene/dataset_readers.py#L191-L197)) is consistent between train and render but it's unknown whether the paper used the same split. Worth confirming with supervisor — if the paper used a different held-out set, that's a separate metric-gap source, not a pixel shift.

## What fixing it would take

Three options, in increasing effort:

1. **Switch to gsplat** for the rasterizer (drop-in `rasterization()` call that accepts `Ks` with cx/cy). This is what splatfacto already does. Biggest gain for least implementation risk; the `flashsplat_rasterization` step would still need its own fix but the train/render/metrics path would be sound.
2. **Patch diff-gaussian-rasterization** to accept cx/cy. Means rebuilding the CUDA submodule with extra args, propagating cx/cy through `GaussianRasterizationSettings`, fixing CUDA projection. Doable but invasive.
3. **Asymmetric projection matrix hack**: read cx/cy, build a non-symmetric frustum in `getProjectionMatrix` so `P[0,2] = -(2*cx/W - 1)` and `P[1,2] = -(2*cy/H - 1)`. The CUDA code applies `projmatrix` for projection, so this can shift the rendered center to match GT without rebuilding the rasterizer. But `tanfovx/tanfovy` is also used internally for tile binning / radii, so there can be subtle binning artifacts at large offsets. Worth trying as a quick test.

Recommended: option 1 if the supervisor is open to a switch, otherwise option 3 as a low-effort experiment to confirm. Even if results aren't perfect, PSNR should jump noticeably on FIP plots after the fix.
