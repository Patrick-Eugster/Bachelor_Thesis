"""Stage A marker localizer, VERSION 4 (hybrid). READ-ONLY on the data.

v4 combines the two earlier approaches, each used for what it is good at:

  * v2 (ellipse on the whole coded pattern): good at FINDING the plates (recall) and rejecting
    canopy (white-surround) and merging a plate's features (size-aware clustering) — but its
    center sometimes landed on a coded arc instead of the middle.
  * v3 (the central fiducial disk): perfect at the CENTER (a solid grey disk with a white dot,
    on white) and near-zero false positives — but its global dark threshold missed most plates.

So v4 does:
  1. v2 region finding   → candidate marker regions (one cluster per plate)
  2. v3 fiducial search, run LOCALLY inside each region → snap the center to the fiducial's
     white dot (the surveyed reference point). Running the threshold locally, inside a known
     plate, sidesteps the global-threshold fragility that wrecked v3's recall.
  3. a region with NO fiducial inside is dropped → extra false-positive rejection.

Result targets all three problems: center always on the fiducial (#1), fewer false positives
(#2, fiducial-confirmation), and v2-level recall (#3, same region finder). When a region has
no detectable fiducial (very distant/occluded) we keep the cluster centroid only if
`keep_without_fiducial` is set, else drop it.

READ-ONLY: writes overlays to marker_vis_v4/ + a JSON to logs/marker_detections_v4.json.
No IDs (Stage B) and no triangulation (Step 2) yet.

Typical usage:
    python src/preprocessing/detect_markers_v4.py field=field_A plot=20250609
    python src/preprocessing/detect_markers_v4.py field=field_A plot=20250609 limit=5

Config: configs/preprocessing/detect_markers_v4.yaml
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


# ---------- v2 part: find marker regions ----------

def fit_candidate_ellipses(bgr, cfg):
    """v2 step 1-2: Canny edges → fit ellipses → keep round, well-fitting ones (candidates)."""
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
        fit_ratio = a / ell_area
        axis_ratio = min(MA, ma) / max(MA, ma)
        if not (cfg.fit_ratio_lo < fit_ratio < cfg.fit_ratio_hi):
            continue
        if axis_ratio < cfg.min_axis_ratio:
            continue
        out.append({"center": (float(cx), float(cy)), "axes": (float(MA), float(ma)), "angle": float(ang)})
    return out


def white_surround_frac(hsv, gray, e, cfg):
    """v2 step 3: fraction of WHITE pixels in a ring outside the ellipse (is it on a plate?).
    "White" is CONTRAST-RELATIVE, not a hardcoded brightness: within the local crop we let Otsu
    pick the bright/dark split, so "bright = plate" adapts to the local sunlight/shadow/exposure
    instead of a fixed value like V≥140 (which breaks across lighting/phone/compression). A light
    saturation ceiling stays (white is desaturated — fairly lighting-robust)."""
    (cx, cy) = e["center"]; (MA, ma) = e["axes"]; ang = e["angle"]
    H, W = hsv.shape[:2]
    r_out = max(MA, ma) * cfg.surround_scale_outer
    x0 = max(0, int(cx - r_out)); y0 = max(0, int(cy - r_out))
    x1 = min(W, int(cx + r_out)); y1 = min(H, int(cy + r_out))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sub = hsv[y0:y1, x0:x1]
    sub_gray = gray[y0:y1, x0:x1]
    ec = (cx - x0, cy - y0)
    mask = np.zeros(sub.shape[:2], np.uint8)
    cv2.ellipse(mask, (ec, (MA * cfg.surround_scale_outer, ma * cfg.surround_scale_outer), ang), 255, -1)
    cv2.ellipse(mask, (ec, (MA * cfg.surround_scale_inner, ma * cfg.surround_scale_inner), ang), 0, -1)
    sel = mask > 0
    n = int(sel.sum())
    if n == 0:
        return 0.0
    # local Otsu split → "bright" relative to THIS region's lighting (not an absolute cutoff)
    otsu, _ = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = sub_gray > otsu
    lowsat = sub[:, :, 1] <= cfg.white_s_max
    white = bright & lowsat
    return float((white & sel).sum()) / float(n)


def cluster_size_aware(items, reach_factor, min_reach):
    """v2 step 4: size-aware union-find clustering (merge reach scales with ellipse size)."""
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
            if d <= max(min_reach, reach_factor * max(si, sj)):
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    clusters = []
    for members in groups.values():
        cx = sum(m["center"][0] for m in members) / len(members)
        cy = sum(m["center"][1] for m in members) / len(members)
        size = max(max(m["axes"]) for m in members)   # biggest feature = ~plate scale
        clusters.append({"cx": cx, "cy": cy, "size": size, "members": members})
    return clusters


# ---------- v3 part: find the fiducial inside a region ----------

def find_fiducial_local(gray, hsv, cx, cy, search_r, cfg):
    """Search for the central fiducial (solid round dark disk + white dot) in a LOCAL window
    around a region center. Local thresholding is robust (the disk is darker than the plate
    right here). Returns (fx, fy, score) or None. score = white_dot fraction (soft confidence)."""
    H, W = gray.shape[:2]
    x0 = max(0, int(cx - search_r)); y0 = max(0, int(cy - search_r))
    x1 = min(W, int(cx + search_r)); y1 = min(H, int(cy + search_r))
    g = gray[y0:y1, x0:x1]
    if g.size == 0:
        return None
    # local Otsu: split this window into dark (disk/arcs) vs bright (plate) automatically.
    # otsu_val is the relative dark/bright boundary — we reuse it for the white-dot test below
    # so "white dot" means "brighter than this region's own split", not a hardcoded value.
    otsu_val, dark = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < cfg.fid_min_area or a > cfg.fid_max_area:
            continue
        (lx, ly), r = cv2.minEnclosingCircle(c)
        if r < cfg.fid_min_radius:
            continue
        circ = a / (np.pi * r * r)
        sol = a / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)
        if circ < cfg.fid_min_circularity or sol < cfg.fid_min_solidity:
            continue
        gx, gy = int(lx), int(ly)            # disk center in the local window
        # white dot at the disk center → refine + score
        ri = max(2, int(r * cfg.center_inner_frac))
        px0 = max(0, gx - ri); py0 = max(0, gy - ri)
        patch = g[py0:gy + ri, px0:gx + ri]
        if patch.size == 0:
            continue
        # white dot = brighter than the local Otsu split (relative), not an absolute cutoff
        bright = patch > otsu_val
        wdot = float(bright.mean())
        if bright.sum() > 0:
            ys, xs = np.nonzero(bright)
            fx = x0 + px0 + float(xs.mean())
            fy = y0 + py0 + float(ys.mean())
        else:
            fx = x0 + float(lx); fy = y0 + float(ly)
        # prefer the disk nearest the region center, weighted by having a white dot
        dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
        cand = {"fx": fx, "fy": fy, "wdot": wdot, "dist": dist, "r": float(r)}
        if best is None or (cand["wdot"] > 0.01 and cand["dist"] < best["dist"]):
            best = cand
    if best is None:
        return None
    return best["fx"], best["fy"], best["wdot"], best["r"]


def detect_one(bgr, cfg):
    """Hybrid: find regions (v2), then snap each to its fiducial (v3 local). Returns markers."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1. v2 candidates on a white plate
    cands = fit_candidate_ellipses(bgr, cfg)
    kept = [e for e in cands if white_surround_frac(hsv, gray, e, cfg) >= cfg.min_white_surround]

    # 2. cluster into one region per plate
    clusters = cluster_size_aware(kept, cfg.cluster_reach_factor, cfg.cluster_min_reach)

    dets = []
    for cl in clusters:
        if len(cl["members"]) < cfg.min_cluster_size:
            continue
        # 3. find the fiducial locally; search radius scales with the region size
        search_r = max(cfg.fid_search_min, cl["size"] * cfg.fid_search_factor)
        fid = find_fiducial_local(gray, hsv, cl["cx"], cl["cy"], search_r, cfg)
        if fid is not None:
            fx, fy, wdot, r = fid
            dets.append({"center": [round(fx, 2), round(fy, 2)], "source": "fiducial",
                         "white_dot": round(wdot, 3), "fid_radius": round(r, 1),
                         "n_ellipses": len(cl["members"])})
        elif cfg.keep_without_fiducial:
            dets.append({"center": [round(cl["cx"], 2), round(cl["cy"], 2)], "source": "centroid",
                         "white_dot": 0.0, "fid_radius": 0.0, "n_ellipses": len(cl["members"])})
        # else: region with no fiducial → dropped (false-positive rejection)
    return dets


def draw_overlay(bgr, dets, max_width):
    """Draw each marker: magenta ring + center dot (green if fiducial-snapped, yellow if centroid)."""
    vis = bgr.copy()
    for i, d in enumerate(dets):
        cx, cy = int(d["center"][0]), int(d["center"][1])
        col = (0, 255, 0) if d["source"] == "fiducial" else (0, 255, 255)
        cv2.circle(vis, (cx, cy), 60, (255, 0, 255), 4)
        cv2.circle(vis, (cx, cy), 9, col, -1)
        cv2.putText(vis, str(i), (cx + 70, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 0, 255), 6)
    H, W = vis.shape[:2]
    if max_width > 0 and W > max_width:
        s = max_width / float(W)
        vis = cv2.resize(vis, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return vis


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/detect_markers_v4")
def main(cfg: DictConfig):
    """Run the hybrid v4 localizer over a plot, writing overlays + a JSON to a v4 folder."""
    print("--- detect_markers_v4 config ---")
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
    print(f"v4 (hybrid) localizing markers in {len(files)} images from {image_dir} ...")
    print(f"Overlays → {vis_dir}/")

    per_image = {}
    counts = []
    n_fid = 0
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(image_dir, f))
        if bgr is None:
            print(f"  [skip] could not decode {f}")
            continue
        dets = detect_one(bgr, cfg)
        per_image[f] = dets
        counts.append(len(dets))
        n_fid += sum(1 for d in dets if d["source"] == "fiducial")
        cv2.imwrite(os.path.join(vis_dir, f), draw_overlay(bgr, dets, cfg.overlay_max_width))
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {f:<32} → {len(dets)} marker(s)")

    counts = np.array(counts) if counts else np.array([0])
    n_at_expected = int(np.sum(counts == cfg.expected_markers))
    n_zero = int(np.sum(counts == 0))
    total = int(counts.sum())

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("      MARKER LOCALIZATION v4 (hybrid) SUMMARY")
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
    print(f"{'  of which fiducial-snapped:':<30} {n_fid} ({100*n_fid/max(1,total):.0f}%)")
    print("-" * 60)
    m, s = divmod(int(elapsed), 60)
    print(f"{'TOTAL TIME:':<30} {m}m {s}s")
    print("=" * 60 + "\n")

    report = {
        "field": cfg.field, "plot": cfg.plot, "image_subdir": cfg.image_subdir,
        "n_images": len(per_image), "expected_markers": cfg.expected_markers,
        "counts": {"min": int(counts.min()), "max": int(counts.max()),
                   "mean": float(counts.mean()), "median": float(np.median(counts)),
                   "n_zero": n_zero, "n_at_expected": n_at_expected, "total": total,
                   "n_fiducial_snapped": n_fid},
        "per_image": per_image, "elapsed_s": elapsed,
    }
    out_path = os.path.join(cfg.source_path, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Detections JSON written to {out_path}\n")
    print("NOTE: v4 is localization only — no IDs, no triangulation, data untouched.\n")


if __name__ == "__main__":
    main()
