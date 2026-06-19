"""Stage A marker localizer, VERSION 5 (template matching / NCC). READ-ONLY on the data.

The v1-v4 heuristics hit a ceiling: every hand-tuned cue (white, ellipse, dark disk) has canopy
lookalikes that pass it and marker exceptions that fail it, so recall + precision + a correct
center can't all be satisfied at once. v5 changes the KIND of method instead of the thresholds.

Idea (Option A from the plan): the central fiducial is the SAME on every marker and is
rotation-invariant — a solid grey disk with a white dot at its exact center, sitting on the white
plate. So instead of describing it with rules, we make a small picture of it (a "template") and
slide it over the image measuring how well it lines up. Where the image really looks like the
fiducial, the match score spikes; the canopy doesn't produce that concentric pattern.

Pipeline per image:
  1. Build a synthetic bullseye template: white plate -> grey disk -> white center dot. We leave
     out the coded ID ring on purpose (it VARIES per marker; the fiducial core does not), so one
     template matches all 6.
  2. Multi-scale NCC: a fiducial is bigger up close, smaller far away, so we match the template at
     a range of disk radii (cv2.matchTemplate, TM_CCOEFF_NORMED). NCC = Normalized Cross-
     Correlation: it divides out local brightness/contrast, so a correct pattern scores ~1.0
     whether the spot is sunny or shaded (lighting-robust by construction).
  3. Non-maximum suppression: one fiducial fires at several nearby positions/scales -> keep only
     the strongest peak per cluster.
  4. Contrast-relative pattern guard: confirm the matched spot's disk really is darker than the
     plate around it (a RELATIVE drop, not an absolute value) -> rejects flat patches where the
     normalized score fired on noise.
  5. White-dot refine: snap the center to the white dot via a local Otsu split (the surveyed
     reference point, sub-pixel) -- reused from v4.

Why this should beat v1-v4: the score reacts to the WHOLE concentric pattern (far more specific
than "roundish dark thing on something white-ish"), and the correlation peak IS the center by
definition (fixes the center drift). A partly occluded fiducial still partially correlates.

READ-ONLY: writes overlays to marker_vis_v5/ + a JSON to logs/marker_detections_v5.json.
No IDs (Stage B) and no triangulation (Step 2) yet.

Typical usage:
    python src/preprocessing/detect_markers_v5.py field=field_A plot=20250609
    python src/preprocessing/detect_markers_v5.py field=field_A plot=20250609 limit=5

Config: configs/preprocessing/detect_markers_v5.yaml
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def build_bullseye_template(disk_radius, cfg):
    """Make a small grayscale picture of the fiducial: white plate -> grey disk -> white center
    dot. The coded ID ring is left out on purpose (it varies per marker; the core does not), so
    this one template matches all 6 markers. Absolute pixel values barely matter because NCC
    normalizes them away — only the relative pattern (plate brighter, disk darker, dot brightest)
    carries the signal."""
    R = int(round(disk_radius))
    half = int(round(R * cfg.template_margin))     # a little white plate margin around the disk
    size = 2 * half + 1
    t = np.full((size, size), cfg.plate_val, np.uint8)         # white plate background
    c = (half, half)
    cv2.circle(t, c, R, cfg.disk_val, -1)                      # solid grey fiducial disk
    dot_r = max(1, int(round(R * cfg.dot_frac)))
    cv2.circle(t, c, dot_r, cfg.dot_val, -1)                   # white center dot
    return t


def match_one_scale(gray, template, ncc_threshold, top_k):
    """Slide one template size over the image (NCC) and return peak candidates above threshold.
    Each candidate is (score, cx, cy) where (cx, cy) is the template-center pixel in the image.
    Caps to the top_k strongest hits so the downstream NMS stays cheap even if many pixels pass."""
    th, tw = template.shape[:2]
    if gray.shape[0] < th or gray.shape[1] < tw:
        return []
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= ncc_threshold)
    if len(ys) == 0:
        return []
    scores = res[ys, xs]
    if top_k and len(scores) > top_k:        # keep only the strongest, drop the long weak tail
        idx = np.argpartition(scores, -top_k)[-top_k:]
        ys, xs, scores = ys[idx], xs[idx], scores[idx]
    off_x, off_y = tw // 2, th // 2          # matchTemplate gives top-left; shift to center
    return [(float(s), float(x + off_x), float(y + off_y)) for s, x, y in zip(scores, xs, ys)]


def nms_peaks(cands, suppress_radius):
    """Greedy non-maximum suppression: sort by score, keep a peak only if it is not within
    suppress_radius of an already-kept (stronger) peak. Collapses the cluster of near-duplicate
    hits one fiducial produces across positions and scales into a single detection."""
    cands = sorted(cands, key=lambda c: c["score"], reverse=True)
    kept = []
    for c in cands:
        ok = True
        for k in kept:
            d = ((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2) ** 0.5
            if d < max(suppress_radius, k["radius"] * 1.5):
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def passes_contrast_guard(gray, cx, cy, radius, cfg):
    """Contrast-RELATIVE sanity check: the matched disk must be darker than the plate ring around
    it by a relative margin. NCC normalizes contrast away, so it can fire on flat/low-contrast
    noise; this rejects those. 'Relative' (drop / plate brightness) means no hardcoded value, so
    it survives sunlight/shadow/phone/compression differences."""
    H, W = gray.shape[:2]
    r = int(round(radius))
    x0 = max(0, int(cx - 2 * r)); y0 = max(0, int(cy - 2 * r))
    x1 = min(W, int(cx + 2 * r)); y1 = min(H, int(cy + 2 * r))
    sub = gray[y0:y1, x0:x1].astype(np.float32)
    if sub.size == 0:
        return False
    lx, ly = cx - x0, cy - y0
    yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
    disk = sub[dist <= r * 0.8]                       # inside the disk (skip the dot-ish core edge)
    plate = sub[(dist >= r * 1.2) & (dist <= r * 1.9)]   # the white ring just outside the disk
    if disk.size == 0 or plate.size == 0:
        return False
    disk_m, plate_m = float(disk.mean()), float(plate.mean())
    rel_drop = (plate_m - disk_m) / (plate_m + 1e-6)
    return rel_drop >= cfg.min_rel_contrast


def white_surround_frac(hsv, gray, cx, cy, radius, cfg):
    """Fraction of WHITE (bright + desaturated) pixels in a ring just outside the disk — i.e. is
    this fiducial sitting on the white plate? This is the strong canopy-rejector from v2-v4: NCC
    finds the bullseye PATTERN, this confirms it's on a PLATE. 'Bright' is a local-Otsu split
    (contrast-relative, not a hardcoded value); only the saturation ceiling is absolute."""
    H, W = gray.shape[:2]
    r_out = radius * cfg.surround_scale_outer
    x0 = max(0, int(cx - r_out)); y0 = max(0, int(cy - r_out))
    x1 = min(W, int(cx + r_out)); y1 = min(H, int(cy + r_out))
    sub = hsv[y0:y1, x0:x1]; sub_gray = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return 0.0
    lx, ly = cx - x0, cy - y0
    yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
    ring = (dist >= radius * cfg.surround_scale_inner) & (dist <= r_out)
    n = int(ring.sum())
    if n == 0:
        return 0.0
    otsu, _ = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = (sub_gray > otsu) & (sub[:, :, 1] <= cfg.white_s_max)
    return float((white & ring).sum()) / float(n)


def refine_to_white_dot(gray, cx, cy, radius, cfg):
    """Snap the center to the white dot inside the disk via a LOCAL Otsu split (the surveyed
    reference point, sub-pixel). Local = robust to lighting; if no bright pixels are found we keep
    the correlation-peak center, which is already on the fiducial. Returns (fx, fy, wdot)."""
    H, W = gray.shape[:2]
    r = max(2, int(round(radius * cfg.dot_search_frac)))
    x0 = max(0, int(cx - r)); y0 = max(0, int(cy - r))
    x1 = min(W, int(cx + r)); y1 = min(H, int(cy + r))
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return cx, cy, 0.0
    otsu, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = patch > otsu
    if bright.sum() == 0:
        return cx, cy, 0.0
    ys, xs = np.nonzero(bright)
    fx = x0 + float(xs.mean()); fy = y0 + float(ys.mean())
    return fx, fy, float(bright.mean())


def detect_one(bgr, templates, work_scale, cfg):
    """Run multi-scale NCC, NMS the peaks, drop flat-noise hits, snap each to its white dot.
    Matching is done on a DOWNSCALED copy (templates are pre-built at that scale) — the bullseye
    pattern is just as detectable small, and matchTemplate is ~1/work_scale² cheaper. The contrast
    guard + white-dot refine then run on the FULL-RES gray, so the reported center is sub-pixel."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if work_scale < 1.0:
        gray_work = cv2.resize(gray_full, None, fx=work_scale, fy=work_scale,
                               interpolation=cv2.INTER_AREA)
    else:
        gray_work = gray_full
    if cfg.match_blur_ksize and cfg.match_blur_ksize >= 3:
        gray_work = cv2.GaussianBlur(gray_work, (cfg.match_blur_ksize, cfg.match_blur_ksize), 0)

    # 1-2. match every template scale on the small image; map hits back to full-res coords + radius
    cands = []
    for radius_full, tmpl in templates:
        for score, cx, cy in match_one_scale(gray_work, tmpl, cfg.ncc_threshold, cfg.top_k_per_scale):
            cands.append({"score": score, "cx": cx / work_scale, "cy": cy / work_scale,
                          "radius": radius_full})
    if not cands:
        return []
    gray = gray_full   # contrast guard + refine use full resolution

    # 3. collapse near-duplicate hits (across position + scale) into one peak per fiducial
    kept = nms_peaks(cands, cfg.nms_suppress_radius)

    dets = []
    for k in kept:
        # 4a. reject flat patches the normalized score fired on (disk not actually darker than plate)
        if not passes_contrast_guard(gray, k["cx"], k["cy"], k["radius"], cfg):
            continue
        # 4b. require the fiducial to sit on the white plate (the strong canopy rejector)
        wsurr = white_surround_frac(hsv, gray, k["cx"], k["cy"], k["radius"], cfg)
        if wsurr < cfg.min_white_surround:
            continue
        # 5. snap to the white dot (sub-pixel surveyed point)
        fx, fy, wdot = refine_to_white_dot(gray, k["cx"], k["cy"], k["radius"], cfg)
        dets.append({"center": [round(fx, 2), round(fy, 2)], "source": "ncc",
                     "score": round(k["score"], 3), "fid_radius": round(k["radius"], 1),
                     "white_surround": round(wsurr, 3), "white_dot": round(wdot, 3)})
    return dets


def draw_overlay(bgr, dets, max_width):
    """Draw each marker: magenta ring + green center dot + index, plus the NCC score as text."""
    vis = bgr.copy()
    for i, d in enumerate(dets):
        cx, cy = int(d["center"][0]), int(d["center"][1])
        cv2.circle(vis, (cx, cy), 60, (255, 0, 255), 4)
        cv2.circle(vis, (cx, cy), 9, (0, 255, 0), -1)
        cv2.putText(vis, f"{i} ({d['score']:.2f})", (cx + 70, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 5)
    H, W = vis.shape[:2]
    if max_width > 0 and W > max_width:
        s = max_width / float(W)
        vis = cv2.resize(vis, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return vis


def build_template_bank(cfg, work_scale):
    """Build one bullseye template per disk radius across the configured (FULL-RES) size range
    (geometric spacing covers near + far fiducials evenly). Each template is drawn at the
    DOWNSCALED radius used for matching, but tagged with its full-res radius for later refining."""
    radii = np.geomspace(cfg.disk_radius_min, cfg.disk_radius_max, cfg.disk_radius_num)
    bank = []
    for r in radii:
        r_work = max(3.0, float(r) * work_scale)   # never let the matched template get too tiny
        bank.append((float(r), build_bullseye_template(r_work, cfg)))
    return bank


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/detect_markers_v5")
def main(cfg: DictConfig):
    """Run the template-matching (NCC) localizer over a plot, writing overlays + a JSON to v5 folders."""
    print("--- detect_markers_v5 config ---")
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

    # derive the matching downscale from the first decodable image (phone images are uniform size)
    first = cv2.imread(os.path.join(image_dir, files[0]))
    if first is None:
        print(f"ERROR: could not decode first image {files[0]}")
        return
    W0 = first.shape[1]
    work_scale = min(1.0, cfg.match_max_width / float(W0)) if cfg.match_max_width > 0 else 1.0
    templates = build_template_bank(cfg, work_scale)
    print(f"Built {len(templates)} template scales, full-res disk radius "
          f"{cfg.disk_radius_min}-{cfg.disk_radius_max} px; matching at {work_scale:.3f}× "
          f"({int(W0*work_scale)} px wide).")

    vis_dir = os.path.join(cfg.source_path, cfg.output_vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"v5 (template matching) localizing markers in {len(files)} images from {image_dir} ...")
    print(f"Overlays → {vis_dir}/")

    per_image = {}
    counts = []
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(image_dir, f))
        if bgr is None:
            print(f"  [skip] could not decode {f}")
            continue
        dets = detect_one(bgr, templates, work_scale, cfg)
        per_image[f] = dets
        counts.append(len(dets))
        cv2.imwrite(os.path.join(vis_dir, f), draw_overlay(bgr, dets, cfg.overlay_max_width))
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {f:<32} → {len(dets)} marker(s)")

    counts = np.array(counts) if counts else np.array([0])
    n_at_expected = int(np.sum(counts == cfg.expected_markers))
    n_zero = int(np.sum(counts == 0))
    total = int(counts.sum())

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("      MARKER LOCALIZATION v5 (template / NCC) SUMMARY")
    print("=" * 60)
    print(f"{'Plot:':<30} {cfg.field}/{cfg.plot}")
    print(f"{'Images processed:':<30} {len(per_image)}")
    print(f"{'Expected markers/image:':<30} {cfg.expected_markers} (not all visible per view)")
    print("-" * 60)
    print(f"{'markers/image  min/max:':<30} {counts.min()} / {counts.max()}")
    print(f"{'markers/image  mean:':<30} {counts.mean():.2f}")
    print(f"{'markers/image  median:':<30} {int(np.median(counts))}")
    print(f"{'images with 0:':<30} {n_zero}")
    print(f"{'images with exactly ' + str(cfg.expected_markers) + ':':<30} {n_at_expected}")
    print(f"{'total markers:':<30} {total}")
    print("-" * 60)
    m, s = divmod(int(elapsed), 60)
    print(f"{'TOTAL TIME:':<30} {m}m {s}s")
    print("=" * 60 + "\n")

    report = {
        "field": cfg.field, "plot": cfg.plot, "image_subdir": cfg.image_subdir,
        "n_images": len(per_image), "expected_markers": cfg.expected_markers,
        "counts": {"min": int(counts.min()), "max": int(counts.max()),
                   "mean": float(counts.mean()), "median": float(np.median(counts)),
                   "n_zero": n_zero, "n_at_expected": n_at_expected, "total": total},
        "per_image": per_image, "elapsed_s": elapsed,
    }
    out_path = os.path.join(cfg.source_path, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Detections JSON written to {out_path}\n")
    print("NOTE: v5 is localization only — no IDs, no triangulation, data untouched.\n")


if __name__ == "__main__":
    main()
