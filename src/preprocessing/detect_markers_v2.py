"""Stage A marker localizer, VERSION 2 (ellipse-based). READ-ONLY on the data.

v1 (detect_markers.py) keyed on the white SQUARE plate + a dark blob, which was too
weak: false positives on bright canopy, misses on tilted/distant plates. v2 keys on the
markers' real geometry instead — they are coded CIRCULAR targets, so their rings project
to ELLIPSES from any angle. The pipeline:

  1. Canny edges → fit an ellipse to each edge contour            (candidate generator)
  2. keep round, well-fitting ellipses                           (drop junk fits)
  3. keep only ellipses sitting on a WHITE plate (white surround) (the strong filter)
  4. cluster surviving ellipses by center                        (one marker = many rings)
  5. per cluster, pick the marker center (the bright fiducial)    (sub-pixel-ish point)

Ellipse-fit alone is not a detector (the canopy makes hundreds of roundish ellipses) — the
white-surround test in step 3 is what separates markers from canopy, and the ellipse gives
a far better center than v1's square.

Like v1 this only WRITES overlays (marker_vis_v2/) + a JSON (logs/marker_detections_v2.json);
it never modifies images/, sparse/, etc. No IDs (Stage B) and no triangulation (Step 2) yet.

Typical usage:
    python src/preprocessing/detect_markers_v2.py field=field_A plot=20250609
    python src/preprocessing/detect_markers_v2.py field=field_A plot=20250609 limit=5

Config: configs/preprocessing/detect_markers_v2.yaml
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def fit_candidate_ellipses(bgr, cfg):
    """Step 1-2: Canny edges → fit an ellipse to each contour, keep round well-fitting ones.
    Returns a list of (ellipse, axis_ratio, fit_ratio) where ellipse = ((cx,cy),(MA,ma),angle).
    This is only a CANDIDATE generator — the canopy produces many roundish ellipses too."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    edges = cv2.Canny(gray, cfg.canny_lo, cfg.canny_hi)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if len(c) < cfg.min_contour_pts:
            continue
        a = cv2.contourArea(c)
        if a < cfg.min_ellipse_area or a > cfg.max_ellipse_area:
            continue
        try:
            e = cv2.fitEllipse(c)
        except cv2.error:
            continue
        (cx, cy), (MA, ma), ang = e
        if MA < 3 or ma < 3:
            continue
        ell_area = np.pi * (MA / 2.0) * (ma / 2.0)
        if ell_area <= 0:
            continue
        fit_ratio = a / ell_area          # contour area vs fitted-ellipse area (≈1 = good fit)
        axis_ratio = min(MA, ma) / max(MA, ma)   # 1 = circle, low = very elongated
        if not (cfg.fit_ratio_lo < fit_ratio < cfg.fit_ratio_hi):
            continue
        if axis_ratio < cfg.min_axis_ratio:
            continue
        out.append((e, round(axis_ratio, 3), round(fit_ratio, 3)))
    return out


def white_surround_frac(hsv, e, cfg):
    """Step 3: fraction of WHITE (bright + desaturated) pixels in a ring just OUTSIDE the
    ellipse — i.e. is this ellipse sitting on a white plate? Real marker rings are; canopy
    ellipses are surrounded by green. Computed on a local crop for speed."""
    (cx, cy), (MA, ma), ang = e
    H, W = hsv.shape[:2]
    r_out = max(MA, ma) * cfg.surround_scale_outer
    # local crop around the ellipse so we don't mask the whole 12 MP image per candidate
    x0 = max(0, int(cx - r_out)); y0 = max(0, int(cy - r_out))
    x1 = min(W, int(cx + r_out)); y1 = min(H, int(cy + r_out))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = hsv[y0:y1, x0:x1]
    ec = (cx - x0, cy - y0)
    mask = np.zeros(sub.shape[:2], np.uint8)
    cv2.ellipse(mask, (ec, (MA * cfg.surround_scale_outer, ma * cfg.surround_scale_outer), ang), 255, -1)
    cv2.ellipse(mask, (ec, (MA * cfg.surround_scale_inner, ma * cfg.surround_scale_inner), ang), 0, -1)
    sel = mask > 0
    n = int(sel.sum())
    if n == 0:
        return 0.0
    V = sub[:, :, 2]; S = sub[:, :, 1]
    white = (V >= cfg.white_v_min) & (S <= cfg.white_s_max)
    return float((white & sel).sum()) / float(n)


def bright_center_score(gray, e):
    """How bright is the very center of the ellipse (the white center dot of the fiducial)?
    Used to pick which ellipse in a cluster is the central fiducial = the marker point."""
    (cx, cy), (MA, ma), ang = e
    H, W = gray.shape[:2]
    r = max(2, int(min(MA, ma) * 0.25))
    x0 = max(0, int(cx - r)); y0 = max(0, int(cy - r))
    x1 = min(W, int(cx + r)); y1 = min(H, int(cy + r))
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float((patch > 150).mean())


def cluster_by_center(items, reach_factor, min_reach):
    """Step 4: SIZE-AWARE clustering of ellipses into one-per-marker. A single marker fits
    many ellipses (fiducial + coded arcs), but how far apart they are in PIXELS depends on
    how close/zoomed the plate is — a hardcoded pixel distance is wrong. So we merge two
    ellipses when their centers are close RELATIVE TO THEIR SIZE: dist ≤ reach_factor * the
    larger ellipse's major axis. This auto-scales with camera distance, zoom and resolution.
    Uses union-find so a fiducial→arc→arc chain all ends up in one cluster."""
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        (xi, yi) = items[i]["center"]; si = max(items[i]["axes"])
        for j in range(i + 1, n):
            (xj, yj) = items[j]["center"]; sj = max(items[j]["axes"])
            d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            # reach scales with the bigger ellipse; min_reach is a small floor for tiny fiducials
            reach = max(min_reach, reach_factor * max(si, sj))
            if d <= reach:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    clusters = []
    for members in groups.values():
        cx = sum(m["center"][0] for m in members) / len(members)
        cy = sum(m["center"][1] for m in members) / len(members)
        clusters.append({"cx": cx, "cy": cy, "members": members})
    return clusters


def detect_one(bgr, cfg):
    """Run the full v2 pipeline on one image. Returns the list of detected markers, each with
    a center, the member-ellipse count, and the scores used — so the JSON is auditable."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1-2. candidate ellipses
    cands = fit_candidate_ellipses(bgr, cfg)

    # 3. keep only those on a white plate
    kept = []
    for e, axis_ratio, fit_ratio in cands:
        ws = white_surround_frac(hsv, e, cfg)
        if ws < cfg.min_white_surround:
            continue
        (cx, cy), (MA, ma), ang = e
        kept.append({
            "center": (float(cx), float(cy)),
            "axes": (float(MA), float(ma)),
            "angle": float(ang),
            "white_surround": round(ws, 3),
            "axis_ratio": axis_ratio,
            "bright_center": round(bright_center_score(gray, e), 3),
        })

    # 4. cluster into one-per-marker (size-aware — scales with how big the plate appears)
    clusters = cluster_by_center(kept, cfg.cluster_reach_factor, cfg.cluster_min_reach)

    dets = []
    for cl in clusters:
        members = cl["members"]
        if len(members) < cfg.min_cluster_size:
            continue
        # 5. marker point = the member with the brightest center (the fiducial); fallback centroid
        fiducial = max(members, key=lambda m: m["bright_center"])
        if fiducial["bright_center"] > 0.05:
            mx, my = fiducial["center"]
        else:
            mx, my = cl["cx"], cl["cy"]
        dets.append({
            "center": [round(mx, 2), round(my, 2)],
            "n_ellipses": len(members),
            "max_white_surround": round(max(m["white_surround"] for m in members), 3),
            "max_bright_center": round(max(m["bright_center"] for m in members), 3),
            "members": [{"center": [round(m["center"][0], 1), round(m["center"][1], 1)],
                         "axes": [round(m["axes"][0], 1), round(m["axes"][1], 1)],
                         "angle": round(m["angle"], 1)} for m in members],
        })
    return dets


def draw_overlay(bgr, dets, max_width):
    """Draw each marker's member ellipses (green) + the chosen center (red dot) + index."""
    vis = bgr.copy()
    for i, d in enumerate(dets):
        for m in d["members"]:
            c = (int(m["center"][0]), int(m["center"][1]))
            axes = (int(m["axes"][0] / 2), int(m["axes"][1] / 2))
            cv2.ellipse(vis, c, axes, m["angle"], 0, 360, (0, 255, 0), 3)
        cx, cy = int(d["center"][0]), int(d["center"][1])
        # big magenta ring so each detection is spottable in the full image (not just when zoomed)
        cv2.circle(vis, (cx, cy), 70, (255, 0, 255), 5)
        cv2.circle(vis, (cx, cy), 10, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (cx + 80, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 6)
    H, W = vis.shape[:2]
    if max_width > 0 and W > max_width:
        s = max_width / float(W)
        vis = cv2.resize(vis, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return vis


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/detect_markers_v2")
def main(cfg: DictConfig):
    """Run the ellipse-based v2 localizer over a plot, writing overlays + a JSON to a v2 folder."""
    print("--- detect_markers_v2 config ---")
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
    print(f"v2 (ellipse) localizing markers in {len(files)} images from {image_dir} ...")
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
            print(f"  [{i+1}/{len(files)}] {f:<32} → {len(dets)} marker(s)")

    counts = np.array(counts) if counts else np.array([0])
    n_at_expected = int(np.sum(counts == cfg.expected_markers))
    n_zero = int(np.sum(counts == 0))

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("      MARKER LOCALIZATION v2 (ellipse) SUMMARY")
    print("=" * 60)
    print(f"{'Plot:':<30} {cfg.field}/{cfg.plot}")
    print(f"{'Images processed:':<30} {len(per_image)}")
    print(f"{'Expected markers/image:':<30} {cfg.expected_markers} (not all visible per view)")
    print("-" * 60)
    print(f"{'markers/image  min/max:':<30} {counts.min()} / {counts.max()}")
    print(f"{'markers/image  mean:':<30} {counts.mean():.2f}")
    print(f"{'markers/image  median:':<30} {int(np.median(counts))}")
    print(f"{'images with 0 markers:':<30} {n_zero}")
    print(f"{'images with exactly ' + str(cfg.expected_markers) + ':':<30} {n_at_expected}")
    print(f"{'total markers:':<30} {int(counts.sum())}")
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
    print("NOTE: v2 is localization only — no IDs, no triangulation, data untouched.\n")


if __name__ == "__main__":
    main()
