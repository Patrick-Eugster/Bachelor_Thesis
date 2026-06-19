"""Stage A marker localizer, VERSION 3 (fiducial-centric). READ-ONLY on the data.

v2 detected the whole coded pattern (fiducial + arcs) as ellipses, then GUESSED which piece
was the center — unreliable (the red dot sometimes landed on an arc), and it needed clean
ellipse fits on the outer ring (fails under occlusion), and it needed clustering (which split
one plate into several rings).

v3 detects the CENTRAL FIDUCIAL directly. The fiducial is a solid, round, dark disk with a
small WHITE DOT at its exact center, sitting on the white plate (the arcs, by contrast, are
curved/elongated — not solid disks). Detecting it fixes all three problems at once:
  * the center is the fiducial BY CONSTRUCTION (never an arc),
  * the "solid round dark disk + white center + white surround" combo is extremely specific
    → almost no canopy false positives,
  * the fiducial is small + central → still visible when a wheat head covers part of the
    outer coded ring,
  * one fiducial per marker → NO clustering needed (the v2 split bug cannot occur).

The precise reported point is the white center dot (sub-pixel), which is the surveyed
reference point — exactly what Step 2 triangulation needs.

READ-ONLY: writes overlays to marker_vis_v3/ + a JSON to logs/marker_detections_v3.json.
No IDs (Stage B) and no triangulation (Step 2) yet.

Typical usage:
    python src/preprocessing/detect_markers_v3.py field=field_A plot=20250609
    python src/preprocessing/detect_markers_v3.py field=field_A plot=20250609 limit=5

Config: configs/preprocessing/detect_markers_v3.yaml
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def white_surround_frac(hsv, cx, cy, r, cfg):
    """Fraction of WHITE (bright + desaturated) pixels in a ring just outside the disk —
    i.e. is this dark disk sitting on a white plate? This is the strong precision gate:
    a canopy shadow blob is surrounded by green, a real fiducial by white."""
    H, W = hsv.shape[:2]
    r_out = int(r * cfg.surround_scale_outer)
    x0 = max(0, cx - r_out); y0 = max(0, cy - r_out)
    x1 = min(W, cx + r_out); y1 = min(H, cy + r_out)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = hsv[y0:y1, x0:x1]
    ec = (cx - x0, cy - y0)
    mask = np.zeros(sub.shape[:2], np.uint8)
    cv2.circle(mask, ec, r_out, 255, -1)
    cv2.circle(mask, ec, int(r * cfg.surround_scale_inner), 0, -1)
    sel = mask > 0
    n = int(sel.sum())
    if n == 0:
        return 0.0
    white = (sub[:, :, 2] >= cfg.white_v_min) & (sub[:, :, 1] <= cfg.white_s_max)
    return float((white & sel).sum()) / float(n)


def white_dot_center(gray, cx, cy, r, cfg):
    """Refine the disk center to the WHITE DOT (the precise reference point). Inside the inner
    part of the disk, take the centroid of the bright pixels (the dot). Returns
    (refined_cx, refined_cy, white_dot_fraction). Falls back to the disk centroid if no dot."""
    H, W = gray.shape[:2]
    ri = max(2, int(r * cfg.center_inner_frac))
    x0 = max(0, cx - ri); y0 = max(0, cy - ri)
    x1 = min(W, cx + ri); y1 = min(H, cy + ri)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return float(cx), float(cy), 0.0
    bright = patch > cfg.white_dot_v_min
    frac = float(bright.mean())
    if bright.sum() == 0:
        return float(cx), float(cy), 0.0
    ys, xs = np.nonzero(bright)
    return x0 + float(xs.mean()), y0 + float(ys.mean()), frac


def detect_one(bgr, cfg):
    """Find the central fiducials in one image. Each accepted fiducial is one marker — no
    clustering. Returns a list of detections with the refined center + the score metrics."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if cfg.use_adaptive:
        # the fiducial disk is darker than the white plate AROUND it, whatever its absolute
        # grey value — so compare each pixel to its local neighborhood instead of a global
        # cutoff. This is robust to lighting/distance (the fixed dark_max missed most disks).
        blk = cfg.adaptive_block if cfg.adaptive_block % 2 == 1 else cfg.adaptive_block + 1
        dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, blk, cfg.adaptive_C)
        # close small gaps so the disk is one solid blob (the white center dot punches a hole)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    else:
        dark = (gray < cfg.dark_max).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < cfg.min_disk_area or a > cfg.max_disk_area:
            continue
        (fx, fy), r = cv2.minEnclosingCircle(c)
        if r < cfg.min_radius:
            continue
        # solid, round disk = the fiducial (arcs are elongated/curved → fail these)
        circ = a / (np.pi * r * r)
        sol = a / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)
        if circ < cfg.min_circularity or sol < cfg.min_solidity:
            continue
        cx, cy = int(fx), int(fy)
        # strong precision gate: must sit on a white plate
        wsur = white_surround_frac(hsv, cx, cy, r, cfg)
        if wsur < cfg.min_white_surround:
            continue
        # refine center to the white dot; white_dot fraction is a soft confidence (not a hard gate)
        rcx, rcy, wdot = white_dot_center(gray, cx, cy, r, cfg)
        if wdot < cfg.min_white_dot:
            continue
        dets.append({
            "center": [round(rcx, 2), round(rcy, 2)],
            "radius": round(float(r), 1),
            "circularity": round(float(circ), 3),
            "solidity": round(float(sol), 3),
            "white_surround": round(float(wsur), 3),
            "white_dot": round(float(wdot), 3),
        })
    return dets


def draw_overlay(bgr, dets, max_width):
    """Draw each fiducial: a green circle on the disk + a red center dot + index."""
    vis = bgr.copy()
    for i, d in enumerate(dets):
        cx, cy = int(d["center"][0]), int(d["center"][1])
        r = int(d["radius"])
        cv2.circle(vis, (cx, cy), max(r, 8), (0, 255, 0), 3)
        cv2.circle(vis, (cx, cy), 60, (255, 0, 255), 4)
        cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (cx + 70, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 0, 255), 6)
    H, W = vis.shape[:2]
    if max_width > 0 and W > max_width:
        s = max_width / float(W)
        vis = cv2.resize(vis, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return vis


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/detect_markers_v3")
def main(cfg: DictConfig):
    """Run the fiducial-centric v3 localizer over a plot, writing overlays + a JSON to a v3 folder."""
    print("--- detect_markers_v3 config ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------")
    t_start = time.time()

    image_dir = os.path.join(cfg.source_path, cfg.image_subdir)
    if not os.path.isdir(image_dir):
        print(f"ERROR: image dir not found: {image_dir}")
        return
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if cfg.limit and cfg.limit > 0:
        files = files[:cfg.limit]
    if not files:
        print(f"ERROR: no images found in {image_dir}")
        return

    vis_dir = os.path.join(cfg.source_path, cfg.output_vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"v3 (fiducial) localizing markers in {len(files)} images from {image_dir} ...")
    print(f"Overlays → {vis_dir}/")

    per_image = {}
    counts = []
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(image_dir, f))
        if bgr is None:
            print(f"  [skip] could not decode {f}")
            continue
        dets = detect_one(bgr, cfg)
        per_image[f] = dets
        counts.append(len(dets))
        cv2.imwrite(os.path.join(vis_dir, f), draw_overlay(bgr, dets, cfg.overlay_max_width))
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {f:<32} → {len(dets)} fiducial(s)")

    counts = np.array(counts) if counts else np.array([0])
    n_at_expected = int(np.sum(counts == cfg.expected_markers))
    n_zero = int(np.sum(counts == 0))

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("      MARKER LOCALIZATION v3 (fiducial) SUMMARY")
    print("=" * 60)
    print(f"{'Plot:':<30} {cfg.field}/{cfg.plot}")
    print(f"{'Images processed:':<30} {len(per_image)}")
    print(f"{'Expected markers/image:':<30} {cfg.expected_markers} (not all visible per view)")
    print("-" * 60)
    print(f"{'fiducials/image  min/max:':<30} {counts.min()} / {counts.max()}")
    print(f"{'fiducials/image  mean:':<30} {counts.mean():.2f}")
    print(f"{'fiducials/image  median:':<30} {int(np.median(counts))}")
    print(f"{'images with 0:':<30} {n_zero}")
    print(f"{'images with exactly ' + str(cfg.expected_markers) + ':':<30} {n_at_expected}")
    print(f"{'total fiducials:':<30} {int(counts.sum())}")
    print("-" * 60)
    m, s = divmod(int(elapsed), 60)
    print(f"{'TOTAL TIME:':<30} {m}m {s}s")
    print("=" * 60 + "\n")

    report = {
        "field": cfg.field, "plot": cfg.plot, "image_subdir": cfg.image_subdir,
        "n_images": len(per_image), "expected_markers": cfg.expected_markers,
        "counts": {"min": int(counts.min()), "max": int(counts.max()),
                   "mean": float(counts.mean()), "median": float(np.median(counts)),
                   "n_zero": n_zero, "n_at_expected": n_at_expected, "total": int(counts.sum())},
        "per_image": per_image, "elapsed_s": elapsed,
    }
    out_path = os.path.join(cfg.source_path, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Detections JSON written to {out_path}\n")
    print("NOTE: v3 is localization only — no IDs, no triangulation, data untouched.\n")


if __name__ == "__main__":
    main()
