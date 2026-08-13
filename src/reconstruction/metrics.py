#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from gaussians.utils.loss_utils import ssim, create_window
from gaussians.lpipsPyTorch import lpips
import cv2
import json

# ROI (plot region) is a buffered convex hull of the projected markers — same shape family as the
# mask-generation ROI (roi_mask.py), just a SMALLER buffer here since it only defines a metric region.
ROI_BUFFER_FRAC = 0.02   # buffer = this fraction of the image short side, grown outward from the hull

# INNER region: a centered crop keeping this fraction of each side, dropping the outer border. This is
# the crop-fair region for comparing arms with different frame sizes (full 4032 pinhole vs ~3900
# undistorted opencv/agisoft): whole-image PSNR unfairly rewards the cropped arms (their border, the
# hardest pixels, is already gone), so we also score a common border-free centre. Approximate (undistortion
# warps each arm slightly differently) — the marker ROI is the exact same-physical-region version.
INNER_FRAC = 0.8


def _ssim_map(img1, img2, window_size=11):
    """Per-pixel SSIM map (same window/constants as loss_utils.ssim, which only returns the mean) so we
    can average SSIM over an arbitrary ROI polygon mask. img1/img2: [1,C,H,W]. Returns [1,C,H,W]."""
    ch = img1.size(-3)
    w = create_window(window_size, ch).type_as(img1)
    if img1.is_cuda:
        w = w.cuda(img1.get_device())
    pad = window_size // 2
    mu1 = F.conv2d(img1, w, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, w, padding=pad, groups=ch)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(img1 * img1, w, padding=pad, groups=ch) - mu1s
    s2 = F.conv2d(img2 * img2, w, padding=pad, groups=ch) - mu2s
    s12 = F.conv2d(img1 * img2, w, padding=pad, groups=ch) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1s + mu2s + C1) * (s1 + s2 + C2))
from tqdm import tqdm
from gaussians.utils.image_utils import psnr
from argparse import ArgumentParser

# Laplacian kernel for the sharpness metric (variance of the Laplacian = standard focus measure).
_LAP_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)

def laplacian_var(img):
    """Sharpness = variance of the Laplacian on the grayscale image (higher = sharper).
    img is [1,3,H,W] in [0,1]; we scale to 0-255 so the number is comparable to cv2.Laplacian.var().
    We report this for render AND gt: the render/gt RATIO says how much high-freq detail the model
    recovered — decoupled from pixel alignment, so (unlike PSNR) it rewards a genuinely sharper render."""
    gray = img.mean(1, keepdim=True) * 255.0
    lap = F.conv2d(gray, _LAP_KERNEL.to(img.device), padding=1)
    return lap.var().item()


def build_marker_ctx(source_path):
    """Parse the COLMAP sparse poses + triangulated 3D markers ONCE for a scene, so we can project the
    markers into each test view. Returns a ctx dict, or None when markers aren't available in this
    frame (Agisoft subfolder / FIP) — in which case the masked passes are simply skipped.
    (marker_points3d.json lives in the COLMAP plot's logs/, so this is COLMAP-only by construction.)"""
    try:
        from mask_generation import roi_mask as _roi   # imports cv2 — may be absent in a train-only env
    except Exception as e:
        print(f"Masked metrics skipped (could not import roi_mask: {e})")
        return None
    sparse = os.path.join(source_path, "sparse", "0")
    cam_txt = os.path.join(sparse, "cameras.txt")
    img_txt = os.path.join(sparse, "images.txt")
    mk_path = os.path.join(source_path, "logs", "marker_points3d.json")
    if not (os.path.isfile(cam_txt) and os.path.isfile(img_txt) and os.path.isfile(mk_path)):
        return None
    cams = _roi._parse_cameras(cam_txt)
    imgs = _roi._parse_images(img_txt)            # {full_name: (R, t, cam_id)}
    by_stem = {os.path.splitext(n)[0]: v for n, v in imgs.items()}
    with open(mk_path) as f:
        marker_xyz = [v["xyz"] for v in json.load(f).get("points3d", {}).values()]
    return {"roi": _roi, "cams": cams, "by_stem": by_stem, "marker_xyz": marker_xyz}


def project_markers(ctx, stem, H, W):
    """Project the 3D markers into ONE view (by image stem). Returns (roi_box, centers, roi_mask):
      roi_box  = (x0,y0,x1,y1) bounding box of the BUFFERED polygon (used for the LPIPS crop).
      centers  = list of (u,v) projected marker plate centres (for the per-plate marker pass).
      roi_mask = boolean HxW mask of the plot region = convex hull of the markers grown outward by
                 ROI_BUFFER_FRAC (same shape family as the mask-generation ROI, smaller buffer).
    (None, None, None) if the pose is missing or not all markers project in-front + in-frame."""
    if ctx is None or stem not in ctx["by_stem"]:
        return None, None, None
    R, t, cid = ctx["by_stem"][stem]
    model, cw, ch, params = ctx["cams"][cid]
    sx, sy = W / cw, H / ch   # cameras.txt is full-res; render/crop is at H,W -> scale (no-op at resolution=1)
    pts, n_inframe = [], 0
    for xyz in ctx["marker_xyz"]:
        uv = ctx["roi"]._project(xyz, R, t, model, params)   # None if behind the camera; else full-res px
        if uv is None:
            continue
        u, v = uv[0] * sx, uv[1] * sy
        pts.append((u, v))
        if 0 <= u < W and 0 <= v < H:
            n_inframe += 1
    # ROBUSTNESS: require ALL triangulated markers to project in-front AND in-frame for this view, so the
    # ROI is always the SAME full marker ring (never a subset → never an inconsistent region). A view
    # missing any marker is skipped for the masked passes rather than scored on a wrong region.
    n_total = len(ctx["marker_xyz"])
    if len(pts) < n_total or n_inframe < n_total:
        return None, None, None
    pts = np.array(pts, np.float32)
    # buffered convex-hull polygon = the plot region (excludes the rectangle's out-of-plot corners)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    hull_i = np.round(hull).astype(np.int32)
    buf = max(1, int(round(ROI_BUFFER_FRAC * min(W, H))))
    roi_mask = ctx["roi"]._roi_keep_region(hull_i, W, H, buf)   # boolean HxW (hull grown by buf px)
    ys, xs = np.where(roi_mask)
    if xs.size == 0:
        return None, None, None
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None, None, None
    return box, pts, roi_mask


def roi_metrics(render, gt, box, mask):
    """ROI metrics over the buffered-polygon MASK (not a rectangle): PSNR + SSIM are averaged over the
    polygon pixels only (so the rectangle's out-of-plot corners don't count); LPIPS + sharpness stay on
    the polygon's bounding-box crop (LPIPS is patch-based and can't be polygon-masked). Returns the same
    5-tuple (psnr,ssim,lpips,sharp_r,sharp_g) as crop_metrics, or None if the region is too small."""
    x0, y0, x1, y1 = box
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    r, g = render[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1]
    m = torch.from_numpy(np.ascontiguousarray(mask[y0:y1, x0:x1])).to(r.device).bool()  # [h,w]
    if int(m.sum()) < 32:
        return None
    # PSNR over polygon pixels: per-pixel MSE (mean over channels), then mean over the mask
    se = ((r - g) ** 2).mean(dim=-3).squeeze(0)          # [h,w]
    mse = se[m].mean()
    psnr_v = float(-10.0 * torch.log10(mse))
    # SSIM over polygon pixels: per-pixel SSIM map (mean over channels), averaged over the mask
    smap = _ssim_map(r, g).mean(dim=-3).squeeze(0)       # [h,w]
    ssim_v = float(smap[m].mean())
    # LPIPS + sharpness on the bbox rectangle (can't polygon-mask a patch metric)
    lp = float(lpips(r, g, net_type='vgg'))
    return (psnr_v, ssim_v, lp, laplacian_var(r), laplacian_var(g))


# NOTE (robust fallback, kept for later — e.g. Agisoft, whose markers aren't in the COLMAP frame):
# a geometry-free version can detect the white marker plates directly in the GT (high V, low S in HSV,
# plate-sized blobs) and take their convex hull as the ROI — no poses needed. Not used by default; the
# projection above is more accurate when markers ARE in-frame (COLMAP).


def crop_metrics(render, gt, box):
    """Compute ALL metrics (PSNR, SSIM, LPIPS, render+gt sharpness) on a rectangular crop of the
    render/gt tensors — so ROI/marker passes report the same metrics as the whole image, not just PSNR.
    box=(x0,y0,x1,y1). Returns a 5-tuple or None if the crop is too small for SSIM/LPIPS (<32 px)."""
    x0, y0, x1, y1 = box
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    r, g = render[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1]
    return (psnr(r, g).item(), ssim(r, g).item(), lpips(r, g, net_type='vgg').item(),
            laplacian_var(r), laplacian_var(g))


def inner_box(H, W, frac=INNER_FRAC):
    """Centered rectangle keeping `frac` of the width and height (drops the outer border evenly).
    The crop-fair region: arms with different frame sizes are scored on a comparable border-free centre
    instead of the whole image, where a smaller (cropped/undistorted) frame gets an unfair PSNR boost."""
    cw, ch = int(round(W * frac)), int(round(H * frac))
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    return (x0, y0, x0 + cw, y0 + ch)


def agg_masked(rows):
    """rows = list of 5-tuples (psnr,ssim,lpips,sharp_r,sharp_g) from crop_metrics. Average into one
    dict with all metrics + the sharpness ratio (or None if no crops were scored)."""
    if not rows:
        return None
    a = np.array(rows, dtype=np.float64)
    mr, mg = float(a[:, 3].mean()), float(a[:, 4].mean())
    return {"PSNR": float(a[:, 0].mean()), "SSIM": float(a[:, 1].mean()), "LPIPS": float(a[:, 2].mean()),
            "sharpness_render": mr, "sharpness_gt": mg,
            "sharpness_ratio": (mr / mg if mg > 0 else 0.0), "n": len(rows)}

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def build_test_names(source_path):
    """Reconstruct the render-index -> test-image-stem map: render.py saves test renders as 00000.png…
    in getTestCameras() order, which is the sorted-by-name test split (dataset_readers sorts then splits
    via the same split_utils). So test_names[k] is the stem for render 000k.png."""
    from wheat_utils import split_utils
    imgs_dir = os.path.join(source_path, "images")
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(imgs_dir)
                   if f.lower().endswith((".jpg", ".png")))
    _, test = split_utils.compute_eval_split(names, pin_test=split_utils.load_pin_test(source_path))
    return test


def evaluate(model_paths, source_path=None):

    # marker projection context (COLMAP-only: None for Agisoft/FIP → masked passes skipped)
    marker_ctx = build_marker_ctx(source_path) if source_path else None
    test_names = build_test_names(source_path) if marker_ctx is not None else None
    if source_path:
        print(f"Masked metrics (ROI + markers): {'ON (markers projected)' if marker_ctx else 'OFF (no markers in this frame)'}")

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                # skip non-render subfolders that eval steps leave in test/ (e.g. "overlay",
                # "segmentation") — they have no renders/+gt/, so they'd crash metrics and
                # (because the try/except is per-scene) drop the whole scene's results.json.
                # This lets metrics be safely re-run after seg/eval, e.g. for a 15k-vs-30k pass.
                if not (renders_dir.is_dir() and gt_dir.is_dir()):
                    continue
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                sharp_r = []   # whole-image render sharpness (Laplacian var)
                sharp_g = []   # whole-image gt sharpness
                roi_rows = []  # ROI (plot region): one crop_metrics 5-tuple per view
                mk_rows = []   # MARKERS: one 5-tuple per projected plate crop (fair, structured content)
                inner_rows = []  # INNER: one crop_metrics 5-tuple per view (centered crop, crop-fair region)

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    sharp_r.append(laplacian_var(renders[idx]))
                    sharp_g.append(laplacian_var(gts[idx]))
                    # INNER (crop-fair, no markers needed): centered fraction of the frame, all metrics
                    H, W = renders[idx].shape[-2:]
                    inm = crop_metrics(renders[idx], gts[idx], inner_box(H, W))
                    if inm is not None:
                        inner_rows.append(inm)
                    # masked passes (COLMAP: project markers into this view, crop, run ALL metrics)
                    if marker_ctx is not None:
                        k = int(os.path.splitext(image_names[idx])[0])   # "00007.png" -> 7
                        stem = test_names[k] if k < len(test_names) else None
                        H, W = renders[idx].shape[-2:]
                        box, centers, roi_mask = (project_markers(marker_ctx, stem, H, W)
                                                  if stem is not None else (None, None, None))
                        if box is not None:
                            roi_m = roi_metrics(renders[idx], gts[idx], box, roi_mask)   # ROI = buffered plot polygon
                            if roi_m is not None:
                                roi_rows.append(roi_m)
                            rad = int(0.03 * min(H, W))                          # per-plate crops
                            for u, v in centers:
                                mm = crop_metrics(renders[idx], gts[idx],
                                                  (max(0, int(u - rad)), max(0, int(v - rad)),
                                                   min(W, int(u + rad)), min(H, int(v + rad))))
                                if mm is not None:
                                    mk_rows.append(mm)

                mean_sr = float(torch.tensor(sharp_r).mean())
                mean_sg = float(torch.tensor(sharp_g).mean())
                sharp_ratio = mean_sr / mean_sg if mean_sg > 0 else 0.0
                roi = agg_masked(roi_rows)
                markers = agg_masked(mk_rows)
                inner = agg_masked(inner_rows)

                print("  WHOLE  : PSNR {:.2f}  SSIM {:.3f}  LPIPS {:.3f}  sharp {:.1%} of GT".format(
                    torch.tensor(psnrs).mean(), torch.tensor(ssims).mean(), torch.tensor(lpipss).mean(), sharp_ratio))
                if inner:   print("  INNER  : PSNR {:.2f}  SSIM {:.3f}  LPIPS {:.3f}  sharp {:.1%} of GT  ({} views)".format(
                    inner["PSNR"], inner["SSIM"], inner["LPIPS"], inner["sharpness_ratio"], inner["n"]))
                if roi:     print("  ROI    : PSNR {:.2f}  SSIM {:.3f}  LPIPS {:.3f}  sharp {:.1%} of GT  ({} views)".format(
                    roi["PSNR"], roi["SSIM"], roi["LPIPS"], roi["sharpness_ratio"], roi["n"]))
                if markers: print("  MARKERS: PSNR {:.2f}  SSIM {:.3f}  LPIPS {:.3f}  sharp {:.1%} of GT  ({} crops)".format(
                    markers["PSNR"], markers["SSIM"], markers["LPIPS"], markers["sharpness_ratio"], markers["n"]))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "sharpness_render": mean_sr,
                                                        "sharpness_gt": mean_sg,
                                                        "sharpness_ratio": sharp_ratio,
                                                        "inner": inner,
                                                        "roi": roi,
                                                        "markers": markers})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            import traceback
            print(f"!!! Unable to compute metrics for model {scene_dir}: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--source_path', '-s', type=str, default=None,
                        help="dataset dir (with sparse/0 + logs/marker_points3d.json) to enable the "
                             "ROI + marker masked passes; omit / Agisoft frame → whole-image only")
    args = parser.parse_args()
    evaluate(args.model_paths, args.source_path)
