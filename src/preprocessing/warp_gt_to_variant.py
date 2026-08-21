"""Warp the finished pinhole/raw GT masks into each SfM variant's undistorted frame.

Why: the GT masks were hand-labelled on the pinhole `images/` (4032x3024 = raw sensor frame). The
opencv/agisoft variants undistort the SAME raw photos differently (smaller, non-linearly warped), so a
GT drawn in the pinhole frame does NOT line up with an opencv/agisoft seg render. This script rebuilds
each variant's exact undistortion from its camera model and re-maps the GT into that frame, so
eval_seg_2d can later compare the variant's 3D-seg output to GT pixel-for-pixel.

It ONLY warps + validates — it runs NO evaluation. It NEVER writes into the source manual_label/ (the
finished GT); warped copies go to <variant>/manual_label/ only, and existing files are never overwritten
unless overwrite=true.

Run:  python src/preprocessing/warp_gt_to_variant.py field=field_A date=20250715
"""
import glob
import json
import os
import shutil
import sys
import time

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# reuse the 3DGS COLMAP model readers (text + binary) so we parse cameras/images exactly like training
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gaussians"))
from scene.colmap_loader import (read_intrinsics_binary, read_intrinsics_text,  # noqa: E402
                                 read_extrinsics_binary, read_extrinsics_text)


class _Cam:
    """Minimal COLMAP camera record (model-agnostic, unlike the 3DGS text reader which only allows
    PINHOLE/SIMPLE_PINHOLE). Mirrors the fields we use from colmap_loader's Camera."""
    def __init__(self, cid, model, width, height, params):
        self.id, self.model, self.width, self.height, self.params = cid, model, width, height, params


def _read_cameras_txt(path):
    """Parse a COLMAP cameras.txt with ANY distortion model (OPENCV/FULL_OPENCV/...). Returns {id: _Cam}."""
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t = line.split()
            cid = int(t[0]); model = t[1]; w = int(t[2]); h = int(t[3])
            params = [float(x) for x in t[4:]]
            cams[cid] = _Cam(cid, model, w, h, params)
    return cams


def _read_cameras(sparse_dir):
    """Read a COLMAP cameras.{bin,txt} from a sparse dir, preferring .bin. The binary reader handles all
    models; for .txt we use the model-agnostic parser above. Returns {cam_id: camera}."""
    b = os.path.join(sparse_dir, "cameras.bin")
    t = os.path.join(sparse_dir, "cameras.txt")
    return read_intrinsics_binary(b) if os.path.exists(b) else _read_cameras_txt(t)


def _read_images(sparse_dir):
    """Read a COLMAP images.{bin,txt} from a sparse dir, preferring .bin. Returns {img_id: Image}."""
    b = os.path.join(sparse_dir, "images.bin")
    t = os.path.join(sparse_dir, "images.txt")
    return read_extrinsics_binary(b) if os.path.exists(b) else read_extrinsics_text(t)


def _K_and_dist(cam):
    """Build the 3x3 intrinsic matrix K and the OpenCV distortion vector for a COLMAP camera,
    dispatching on its model. Returns (K, dist) where dist is the cv2 coeff array for that model."""
    p = [float(x) for x in cam.params]
    m = cam.model
    if m == "SIMPLE_PINHOLE":                       # f, cx, cy
        fx = fy = p[0]; cx, cy = p[1], p[2]; dist = np.zeros(4)
    elif m == "PINHOLE":                            # fx, fy, cx, cy
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]; dist = np.zeros(4)
    elif m == "SIMPLE_RADIAL":                      # f, cx, cy, k1
        fx = fy = p[0]; cx, cy = p[1], p[2]; dist = np.array([p[3], 0, 0, 0], float)
    elif m == "RADIAL":                             # f, cx, cy, k1, k2
        fx = fy = p[0]; cx, cy = p[1], p[2]; dist = np.array([p[3], p[4], 0, 0], float)
    elif m == "OPENCV":                             # fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]; dist = np.array(p[4:8], float)
    elif m == "FULL_OPENCV":                        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]; dist = np.array(p[4:12], float)
    else:
        raise ValueError(f"unsupported camera model for GT warp: {m}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    return K, dist


def _find_by_stem(images, stem):
    """Find the COLMAP image whose filename contains `stem` (handles Agisoft's _<seq> rename).
    Returns (image_name, camera_id) or (None, None)."""
    for im in images.values():
        if stem in os.path.splitext(os.path.basename(im.name))[0]:
            return im.name, im.camera_id
    return None, None


def _load_gray(path):
    """Load an image as a uint8 grayscale numpy array (or None if missing)."""
    if not path or not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("L"))


def _phase_shift(a, b):
    """Signed sub-pixel translation (dx,dy) between two same-size grayscale images (cv2.phaseCorrelate)."""
    (dx, dy), _ = cv2.phaseCorrelate(np.float32(a), np.float32(b))
    return dx, dy


def _phase_offset(a, b):
    """Magnitude of the sub-pixel shift between two same-size grayscale images (0 = perfectly aligned)."""
    dx, dy = _phase_shift(a, b)
    return float(np.hypot(dx, dy))


def _max_quadrant_offset(a, b):
    """Max phase-correlation offset over the 4 image quadrants. A pure translation gives a small offset in
    all four; a residual scale/warp makes opposite quadrants disagree, so this catches non-translation
    misalignment that a single global offset would miss (SSIM can't be used — it's appearance-confounded
    across pipelines that re-encode the image, e.g. Agisoft)."""
    H, W = a.shape
    hy, hx = H // 2, W // 2
    quads = [(0, hy, 0, hx), (0, hy, hx, W), (hy, H, 0, hx), (hy, H, hx, W)]
    return float(max(_phase_offset(a[y0:y1, x0:x1], b[y0:y1, x0:x1]) for (y0, y1, x0, x1) in quads))


def _largest_submodel(sparse_root):
    """Pick the COLMAP sub-model with the most images under sparse_root (0/, 1/, ...). The mapper can
    spawn a stray small sub-model (e.g. 3 images) alongside the real one, and run_colmap undistorts the
    LARGEST — but the leftover distorted/sparse/0 may be the stray, so hardcoding /0 reads the wrong model.
    Returns the sub-dir path, or sparse_root itself if it has no numbered sub-dirs."""
    subs = sorted(glob.glob(os.path.join(sparse_root, "*")))
    subs = [s for s in subs if os.path.isdir(s) and os.path.basename(s).isdigit()]
    if not subs:
        return sparse_root
    return max(subs, key=lambda s: len(_read_images(s)))


def _variant_camera_maps(base, variant, agisoft_distorted_dir):
    """Resolve, for a variant, the (distorted cameras, undistorted cameras, distorted images,
    undistorted images) needed to build a per-image undistortion. Returns a dict, or None if a
    required file is missing (variant is then skipped)."""
    if variant == "opencv":
        dist_sparse = _largest_submodel(os.path.join(base, "opencv", "distorted", "sparse"))
        und_sparse = os.path.join(base, "opencv", "sparse", "0")
        if not (os.path.isdir(dist_sparse) and os.path.isdir(und_sparse)):
            return None
        return {"dist_cams": _read_cameras(dist_sparse), "und_cams": _read_cameras(und_sparse),
                "dist_imgs": _read_images(dist_sparse), "und_imgs": _read_images(und_sparse),
                "rename": False}
    if variant == "agisoft":
        dist_sparse = os.path.join(agisoft_distorted_dir, "sparse", "0")
        und_sparse = os.path.join(base, "agisoft", "sparse", "0")
        if not (os.path.isdir(dist_sparse) and os.path.isdir(und_sparse)):
            return None
        return {"dist_cams": _read_cameras(dist_sparse), "und_cams": _read_cameras(und_sparse),
                "dist_imgs": _read_images(dist_sparse), "und_imgs": _read_images(und_sparse),
                "rename": True}
    return None


def _warp_instance_gt(src_label, out_label, stem, out_stem, mapx, mapy, applied_shift, overwrite):
    """Warp the per-head INSTANCE map (uint16 ids) into the variant frame with the SAME undistortion maps
    the binary GT used, NEAREST so head ids are never blended. Writes <out>/{out_stem}_sets/ + copies the
    manifest so eval_masks_instance reads the same active set. Never overwrites unless overwrite=true.
    Returns a short status string (does not modify the source _sets/)."""
    src_sets = os.path.join(src_label, f"{stem}_sets")
    man_path = os.path.join(src_sets, "manifest.json")
    if not os.path.exists(man_path):
        return "no_instance_gt"
    man = json.load(open(man_path))
    entry = next((e for e in man.get("sets", []) if e["name"] == man.get("active")), None) \
        or (man.get("sets") or [None])[0]
    if entry is None:
        return "no_active_set"
    inst_png = os.path.join(src_sets, f"{entry['file']}_instances.png")
    inst = cv2.imread(inst_png, cv2.IMREAD_UNCHANGED)
    if inst is None:
        return "instance_png_unreadable"
    if inst.ndim == 3:
        inst = inst[..., 0]
    # same remap as the binary GT (NEAREST keeps integer ids intact), then the same post-shift if one was applied
    w = cv2.remap(inst, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if applied_shift != [0.0, 0.0]:
        M = np.float32([[1, 0, applied_shift[0]], [0, 1, applied_shift[1]]])
        w = cv2.warpAffine(w, M, (w.shape[1], w.shape[0]), flags=cv2.INTER_NEAREST)
    out_sets = os.path.join(out_label, f"{out_stem}_sets")
    out_inst = os.path.join(out_sets, f"{entry['file']}_instances.png")
    if os.path.exists(out_inst) and not overwrite:
        return "exists_skipped"
    os.makedirs(out_sets, exist_ok=True)
    cv2.imwrite(out_inst, w.astype(np.uint16))                 # 16-bit PNG preserves the ids
    shutil.copy(man_path, os.path.join(out_sets, "manifest.json"))
    ids = np.unique(w)
    return f"written({int((ids != 0).sum())} heads)"


def process_variant(base, variant, gt_masks, cfg):
    """Warp every finished GT mask into one variant's undistorted frame + validate. Returns a report
    dict. Writes warped masks to <base>/<variant>/manual_label/ (never the source), overlays + report
    to <base>/<variant>/logs/warp_gt/."""
    # agisoft_distorted_dir is given relative to the repo root (the local-only demoanlage dump), or absolute
    agi_dist = str(cfg.agisoft_distorted_dir)
    maps = _variant_camera_maps(base, variant, agi_dist)
    rows = []
    if maps is None:
        print(f"  [{variant}] missing sparse/camera files -> SKIP")
        return {"variant": variant, "status": "skipped_missing_cameras", "items": []}

    out_label = os.path.join(base, variant, "manual_label")
    src_label = os.path.join(base, "manual_label")
    # HARD GUARD: never write into the source GT folder
    if os.path.realpath(out_label) == os.path.realpath(src_label):
        raise RuntimeError(f"refusing to write into the SOURCE manual_label ({src_label})")
    log_dir = os.path.join(base, variant, "logs", "warp_gt")
    os.makedirs(out_label, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for gt_path in gt_masks:
        stem = os.path.basename(gt_path).replace("_gt_mask.png", "")
        gt = _load_gray(gt_path)  # raw/pinhole frame (4032x3024)

        # map the GT stem to this variant's distorted + undistorted camera
        dist_name, dist_cid = _find_by_stem(maps["dist_imgs"], stem)
        und_name, und_cid = _find_by_stem(maps["und_imgs"], stem)
        if und_cid is None or dist_cid is None:
            print(f"  [{variant}] {stem}: no matching image in variant sparse -> skip")
            rows.append({"stem": stem, "status": "no_variant_image"})
            continue
        out_stem = os.path.splitext(os.path.basename(und_name))[0] if maps["rename"] else stem
        out_path = os.path.join(out_label, f"{out_stem}_gt_mask.png")
        # skip the whole image only when the binary is done AND we're not additionally warping instances;
        # if warp_instances is on we still proceed (binary is re-saved only when missing, never overwritten)
        binary_done = os.path.exists(out_path) and not cfg.overwrite
        if binary_done and not cfg.get("warp_instances", False):
            print(f"  [{variant}] {out_stem}: target exists -> skip (overwrite=false)")
            rows.append({"stem": stem, "out_stem": out_stem, "status": "exists_skipped"})
            continue

        K_dist, dist = _K_and_dist(maps["dist_cams"][dist_cid])
        und_cam = maps["und_cams"][und_cid]
        K_und, _ = _K_and_dist(und_cam)
        Wu, Hu = int(und_cam.width), int(und_cam.height)

        # raw image (= pinhole/input_uniform frame, verified pixel-identical) + the pipeline's REAL
        # undistorted image, used to VALIDATE that our rebuilt undistortion matches the pipeline's.
        raw_img_path = next((os.path.join(base, "images", f"{stem}{e}")
                             for e in (".jpg", ".png", ".JPG", ".jpeg")
                             if os.path.exists(os.path.join(base, "images", f"{stem}{e}"))), None)
        real_und = _load_gray(os.path.join(base, variant, "images", und_name))
        raw = _load_gray(raw_img_path)

        def _build(Ku):
            mx, my = cv2.initUndistortRectifyMap(K_dist, dist, np.eye(3), Ku, (Wu, Hu), cv2.CV_32FC1)
            ou = cv2.remap(raw, mx, my, cv2.INTER_LINEAR) if raw is not None else None
            return mx, my, ou

        # analytic undistortion (centered K) -> warp the GT (NEAREST keeps the mask crisp / label-safe)
        mapx, mapy, our_und = _build(K_und)
        warped = cv2.remap(gt, mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        applied_shift = [0.0, 0.0]
        offset = None
        if our_und is not None and real_und is not None:
            if real_und.shape != our_und.shape:
                real_und = cv2.resize(real_und, (our_und.shape[1], our_und.shape[0]))
            # Agisoft re-centers the principal point -> a pure translation vs our undistortion. Solve it as a
            # POST-undistortion pixel shift (NOT a K.cx/cy nudge, which in a nonlinear undistort adds a
            # spurious scale). opencv/radial already match to ~0 px so this never triggers there.
            if cfg.auto_translation_fix and _phase_offset(our_und, real_und) > float(cfg.max_offset_px):
                tx, ty = _phase_shift(our_und, real_und)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                sz = (our_und.shape[1], our_und.shape[0])
                our_und = cv2.warpAffine(our_und, M, sz)
                warped = cv2.warpAffine(warped, M, sz, flags=cv2.INTER_NEAREST)
                applied_shift = [tx, ty]
            # geometric gate = worst quadrant offset (catches any residual scale/warp, not just translation)
            offset = _max_quadrant_offset(our_und, real_und)
        # SSIM is reported for information ONLY (appearance-confounded across Agisoft's re-encode) — NOT gated
        ssim_val = float(ssim(our_und, real_und, data_range=255)) \
            if (our_und is not None and real_und is not None) else None
        ok = (offset is not None and offset <= float(cfg.max_offset_px))
        flag = "OK" if ok else ("NO_VALIDATION" if offset is None else "OFFSET_HIGH")

        # overlay (always, for eyeballing head alignment)
        if real_und is not None:
            real_bgr = cv2.cvtColor(real_und, cv2.COLOR_GRAY2BGR)
            cont, _ = cv2.findContours((warped >= 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(real_bgr, cont, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(log_dir, f"{out_stem}_overlay.jpg"), real_bgr)

        # the GT only enters manual_label if it PASSED validation; a failed warp stays in logs (REJECTED)
        inst_status = None
        if ok:
            if not binary_done:                    # never re-write an existing binary GT
                Image.fromarray(warped).save(out_path)
            dest = out_path
            # optionally warp the per-head INSTANCE map too (needed for eval_masks_instance on the variant)
            if cfg.get("warp_instances", False):
                inst_status = _warp_instance_gt(src_label, out_label, stem, out_stem,
                                                mapx, mapy, applied_shift, cfg.overwrite)
                print(f"    [{variant}] {out_stem}: instance GT -> {inst_status}")
        else:
            dest = os.path.join(log_dir, f"{out_stem}_gt_mask_REJECTED.png")
            Image.fromarray(warped).save(dest)
        print(f"  [{variant}] {out_stem}: {warped.shape[1]}x{warped.shape[0]}  "
              f"offset={offset if offset is None else round(offset,2)}px  "
              f"ssim={ssim_val if ssim_val is None else round(ssim_val,3)}  "
              f"shift={[round(s,1) for s in applied_shift]}  -> {flag}"
              + ("" if ok else "   (REJECTED -> logs only, NOT in manual_label)"))
        rows.append({"stem": stem, "out_stem": out_stem, "status": "written" if ok else "rejected",
                     "out_path": dest, "size": [Wu, Hu], "dist_model": maps["dist_cams"][dist_cid].model,
                     "offset_px": offset, "ssim": ssim_val, "applied_shift": applied_shift, "validation": flag,
                     "instance_gt": inst_status})

    report = {"variant": variant, "items": rows}
    with open(os.path.join(log_dir, "warp_gt_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


@hydra.main(version_base=None, config_path="../../configs/preprocessing", config_name="warp_gt")
def main(cfg):
    """Warp the finished GT masks into each requested variant's frame + validate. No evaluation is run."""
    t0 = time.time()
    print(OmegaConf.to_yaml(cfg))
    base = os.path.join(cfg.dataset.input_dir, cfg.field, str(cfg.date))
    gt_masks = sorted(glob.glob(os.path.join(base, "manual_label", "*_gt_mask.png")))
    if not gt_masks:
        print(f"No finished GT masks in {base}/manual_label/ — nothing to warp.")
        return
    print(f"Found {len(gt_masks)} finished GT mask(s): {[os.path.basename(g) for g in gt_masks]}")

    reports = [process_variant(base, v, gt_masks, cfg) for v in cfg.variants]

    # boxed summary
    print("\n" + "=" * 64)
    print(f" GT WARP SUMMARY — {cfg.field}/{cfg.date}   ({time.time()-t0:.1f}s)")
    print("=" * 64)
    for r in reports:
        items = r.get("items", [])
        written = [i for i in items if i.get("status") == "written"]
        rejected = [i for i in items if i.get("status") == "rejected"]
        print(f" {r['variant']:8s}: {len(written)} written, {len(rejected)} rejected"
              + (f"  ({r.get('status')})" if not items else ""))
        for i in written + rejected:
            tag = "" if i["status"] == "written" else "  REJECTED(logs only)"
            print(f"    - {i['out_stem']}  {i['size'][0]}x{i['size'][1]}  {i['dist_model']}  "
                  f"offset={i['offset_px'] and round(i['offset_px'],2)}px "
                  f"ssim={i['ssim'] and round(i['ssim'],3)} shift={[round(s,1) for s in i.get('applied_shift',[0,0])]}{tag}")
    print("=" * 64)
    print("Warped GT + overlays under <base>/<variant>/logs/warp_gt/ ; masks in <variant>/manual_label/.")
    print("NO evaluation was run. Source manual_label/ was NOT modified.")


if __name__ == "__main__":
    main()
