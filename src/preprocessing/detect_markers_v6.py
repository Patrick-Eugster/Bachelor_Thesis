"""Stage A marker localizer, VERSION 6 (REAL-template NCC + fiducial-snap). READ-ONLY on the data.

v6 fixes the two problems v5 still had:

  * v5 matched a SYNTHETIC bullseye (drawn in code) → too crude, real fiducials only reached NCC
    ~0.73 (barely above canopy ~0.6). v6 matches a patch of REAL fiducial pixels (cropped once by
    make_fiducial_template.py) → far more discriminating, wider margin, peak lands on the fiducial.
  * v5's center refine searched for the BRIGHTEST pixels near the peak → the white PLATE is the
    brightest thing, so it dragged the center off the fiducial. v6 instead snaps the center with
    v4's find_fiducial_local: locate the round DARK disk first (plate excluded by construction),
    then the white dot INSIDE that disk → correct sub-pixel center.

Pipeline per image:
  1. Build a template bank by resizing the ONE real fiducial crop to a range of disk radii
     (a fiducial is bigger up close, smaller far away — matchTemplate is not scale-invariant).
  2. Multi-scale NCC on a downscaled copy (speed); map peaks back to full-res coords.
  3. NMS → one peak per fiducial.
  4. Two gates (both contrast-relative, lighting-robust): the disk must actually be darker than
     the plate (contrast guard), and the fiducial must sit on the white plate (white-surround).
  5. find_fiducial_local → snap the center to the white dot inside the dark disk (v4's method).

READ-ONLY: writes overlays to marker_vis_v6/ + logs/marker_detections_v6.json. No IDs, no triangulation.

Prereq: run make_fiducial_template.py first to create the real template (default path below).

Usage:
    python src/preprocessing/detect_markers_v6.py field=field_A plot=20250609
    python src/preprocessing/detect_markers_v6.py field=field_A plot=20250609 limit=5

Config: configs/preprocessing/detect_markers_v6.yaml
"""

import json
import math
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def _warp_oblique(t, ratio, theta_deg):
    """Squash a square template by `ratio` along the direction `theta_deg` (in-plane), to simulate a
    frontal marker seen from an oblique angle (a circle foreshortens to an ellipse). The compression
    axis is rotated to theta so we can cover any slant direction. borderValue = template mean so the
    constant fill contributes ~nothing to the mean-subtracted NCC (TM_CCOEFF_NORMED)."""
    h, w = t.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    # anisotropic scale by `ratio` along direction theta: R(theta) @ diag(1,ratio) @ R(-theta)
    A = np.array([[c, -s], [s, c]]) @ np.array([[1.0, 0.0], [0.0, ratio]]) @ np.array([[c, s], [-s, c]])
    M = np.zeros((2, 3), np.float64)
    M[:, :2] = A
    M[:, 2] = [cx - (A[0, 0] * cx + A[0, 1] * cy), cy - (A[1, 0] * cx + A[1, 1] * cy)]
    return cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_AREA, borderValue=float(t.mean()))


def build_template_bank(cfg, work_scale):
    """Resize the ONE real fiducial crop to each disk radius in the bank (geometric spacing).
    The crop has a known canonical disk radius (cfg.canon_radius); to get a template whose disk
    radius is `r` full-res, we want `r*work_scale` work-px, i.e. resize the crop by that / canon.
    If cfg.oblique_templates is on, each scale ALSO gets affine-warped (foreshortened) copies so the
    matcher can find plates viewed at steep oblique angles (late-season tall canopy → circle projects
    to an ellipse the fronto-parallel template misses). Opt-in: it multiplies the template count (and
    the matchTemplate cost) by 1 + len(ratios)*len(rotations) — see docs/preprocessing/markers/MARKER_DETECTOR_LATE_SEASON.md."""
    tmpl_path = os.path.join(hydra.utils.get_original_cwd(), cfg.template_image)
    canon = cv2.imread(tmpl_path, cv2.IMREAD_GRAYSCALE)
    if canon is None:
        raise FileNotFoundError(f"real template not found: {tmpl_path} "
                                f"(run make_fiducial_template.py first)")
    radii = np.geomspace(cfg.disk_radius_min, cfg.disk_radius_max, cfg.disk_radius_num)
    bank = []
    for r in radii:
        factor = (float(r) * work_scale) / float(cfg.canon_radius)
        size = max(7, int(round(canon.shape[0] * factor)))   # keep templates a few px minimum
        bank.append((float(r), cv2.resize(canon, (size, size), interpolation=cv2.INTER_AREA)))
    if cfg.get("oblique_templates", False):
        ratios = list(cfg.get("oblique_ratios", [0.6, 0.42]))
        rots = list(cfg.get("oblique_rotations_deg", [0, 45, 90, 135]))
        for r, t in list(bank):                       # keep the radius; warp the image
            for ra in ratios:
                for th in rots:
                    bank.append((r, _warp_oblique(t, float(ra), float(th))))
    return bank


def match_one_scale(gray, template, ncc_threshold, top_k):
    """Slide one template size over the image (NCC) and return peak candidates above threshold.
    Caps to the top_k strongest hits so the downstream NMS stays cheap."""
    th, tw = template.shape[:2]
    if gray.shape[0] < th or gray.shape[1] < tw:
        return []
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= ncc_threshold)
    if len(ys) == 0:
        return []
    scores = res[ys, xs]
    if top_k and len(scores) > top_k:
        idx = np.argpartition(scores, -top_k)[-top_k:]
        ys, xs, scores = ys[idx], xs[idx], scores[idx]
    off_x, off_y = tw // 2, th // 2          # matchTemplate gives top-left; shift to center
    return [(float(s), float(x + off_x), float(y + off_y)) for s, x, y in zip(scores, xs, ys)]


def nms_peaks(cands, suppress_radius):
    """Greedy non-maximum suppression: keep a peak only if it is not within suppress_radius (or
    1.5× the kept radius) of an already-kept stronger peak — collapses the per-fiducial cluster."""
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
    """Contrast-RELATIVE check: the matched disk must be darker than the plate ring around it by a
    relative margin. NCC normalizes contrast away so it can fire on flat noise; this rejects that."""
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
    disk = sub[dist <= r * 0.8]
    plate = sub[(dist >= r * 1.2) & (dist <= r * 1.9)]
    if disk.size == 0 or plate.size == 0:
        return False
    rel_drop = (float(plate.mean()) - float(disk.mean())) / (float(plate.mean()) + 1e-6)
    return rel_drop >= cfg.min_rel_contrast


def white_surround_frac(hsv, gray, cx, cy, radius, cfg):
    """Fraction of PLATE pixels in a ring just outside the disk — is the fiducial on the marker
    plate? The strong canopy rejector. Two modes (cfg.plate_gate):
      "white"  (default, legacy): plate = BRIGHT (local-Otsu split, contrast-relative) AND
                desaturated (S <= white_s_max). Brittle in late season — a grey/shaded plate next
                to bright sunlit straw fails the brightness test because Otsu locks onto the straw.
      "lowsat" (brightness-invariant): plate = ACHROMATIC (S <= plate_s_max), no brightness test.
                The marker plate is neutral grey/white at ANY light level; wheat is golden/saturated.
                Recovers the late-season misses the white test drops (see config note)."""
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
    if str(cfg.get("plate_gate", "white")) == "lowsat":
        # brightness-invariant: plate is just the low-saturation (achromatic) region, no Otsu bar
        plate = sub[:, :, 1] <= cfg.get("plate_s_max", 110)
        return float((plate & ring).sum()) / float(n)
    otsu, _ = cv2.threshold(sub_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = (sub_gray > otsu) & (sub[:, :, 1] <= cfg.white_s_max)
    return float((white & ring).sum()) / float(n)


def find_fiducial_local(gray, cx, cy, search_r, cfg):
    """v4's fiducial snap: in a LOCAL window, locate the round solid DARK disk (plate excluded by
    construction), then the white dot INSIDE it → sub-pixel center. Returns (fx, fy, wdot) or None.
    This is what fixes v5's center drift (v5 grabbed the bright plate; this grabs the dark disk)."""
    H, W = gray.shape[:2]
    x0 = max(0, int(cx - search_r)); y0 = max(0, int(cy - search_r))
    x1 = min(W, int(cx + search_r)); y1 = min(H, int(cy + search_r))
    g = gray[y0:y1, x0:x1]
    if g.size == 0:
        return None
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
        # reject blobs whose center is far from the NCC peak — the real-template peak is already on
        # the fiducial, so the right disk is the NEAR one; a far dark blob is an arc/neighbor.
        disk_dist = ((x0 + lx - cx) ** 2 + (y0 + ly - cy) ** 2) ** 0.5
        if disk_dist > r * cfg.fid_max_center_offset:
            continue
        # ROBUST center: fit an ellipse to the disk contour and take its center. This handles a
        # partly-occluded disk and a faint/missing white dot far better than a bright-pixel centroid
        # (which a single bright occluding wheat edge could drag off-center). The white dot sits at
        # the disk's geometric center by design, so the ellipse center IS the surveyed point.
        if len(c) >= 5:
            (ex, ey), _, _ = cv2.fitEllipse(c)
            fx, fy = x0 + float(ex), y0 + float(ey)
        else:
            fx, fy = x0 + float(lx), y0 + float(ly)
        # white-dot fraction near the center — kept only as a soft confidence for the JSON
        ri = max(2, int(r * cfg.center_inner_frac))
        gx, gy = int(round(fx - x0)), int(round(fy - y0))
        px0 = max(0, gx - ri); py0 = max(0, gy - ri)
        patch = g[py0:gy + ri, px0:gx + ri]
        wdot = float((patch > otsu_val).mean()) if patch.size else 0.0
        # quality = how disk-like (round × solid); prefer the most disk-like blob near the peak
        quality = circ * sol
        cand = {"fx": fx, "fy": fy, "wdot": wdot, "quality": quality}
        if best is None or cand["quality"] > best["quality"]:
            best = cand
    if best is None:
        return None
    return best["fx"], best["fy"], best["wdot"]


def dedup_by_center(dets, factor):
    """Merge detections whose FINAL (snapped) centers coincide — keep the higher-scoring one.
    Needed because NMS runs on the raw NCC peaks (pre-snap): two peaks from different scales on the
    same marker can survive NMS, then both snap onto the same fiducial → duplicate dots. This second
    pass, run AFTER snapping, collapses them."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        ok = True
        for k in kept:
            dd = ((d["center"][0] - k["center"][0]) ** 2 + (d["center"][1] - k["center"][1]) ** 2) ** 0.5
            if dd < factor * max(d["fid_radius"], k["fid_radius"]):
                ok = False
                break
        if ok:
            kept.append(d)
    return kept


def detect_one(bgr, templates, work_scale, cfg):
    """Real-template multi-scale NCC → NMS → contrast + white-plate gates → fiducial-snap center."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if work_scale < 1.0:
        gray_work = cv2.resize(gray, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    else:
        gray_work = gray
    if cfg.match_blur_ksize and cfg.match_blur_ksize >= 3:
        gray_work = cv2.GaussianBlur(gray_work, (cfg.match_blur_ksize, cfg.match_blur_ksize), 0)

    # 1-2. match every template scale on the small image; map hits to full-res coords + radius
    cands = []
    for radius_full, tmpl in templates:
        for score, cx, cy in match_one_scale(gray_work, tmpl, cfg.ncc_threshold, cfg.top_k_per_scale):
            cands.append({"score": score, "cx": cx / work_scale, "cy": cy / work_scale,
                          "radius": radius_full})
    if not cands:
        return []

    # 3. collapse near-duplicate hits into one peak per fiducial
    kept = nms_peaks(cands, cfg.nms_suppress_radius)

    dets = []
    for k in kept:
        # 4a. reject flat patches (disk not actually darker than plate)
        if not passes_contrast_guard(gray, k["cx"], k["cy"], k["radius"], cfg):
            continue
        # 4b. require the fiducial to sit on the white plate
        wsurr = white_surround_frac(hsv, gray, k["cx"], k["cy"], k["radius"], cfg)
        if wsurr < cfg.min_white_surround:
            continue
        # 5. snap the center to the white dot inside the dark disk (v4 method); else keep NCC peak
        search_r = max(cfg.fid_search_min, k["radius"] * cfg.fid_search_factor)
        fid = find_fiducial_local(gray, k["cx"], k["cy"], search_r, cfg)
        if fid is not None:
            fx, fy, wdot = fid; source = "fiducial"
        elif cfg.keep_without_fiducial:
            fx, fy, wdot = k["cx"], k["cy"], 0.0; source = "ncc"
        else:
            continue   # NCC peak with no real dark disk inside → canopy false positive, drop it
        dets.append({"center": [round(fx, 2), round(fy, 2)], "source": source,
                     "score": round(k["score"], 3), "fid_radius": round(k["radius"], 1),
                     "white_surround": round(wsurr, 3), "white_dot": round(wdot, 3)})
    # collapse any duplicates that snapped to the same fiducial (NMS ran pre-snap)
    return dedup_by_center(dets, cfg.dedup_radius_factor)


def draw_overlay(bgr, dets, max_width):
    """Draw each marker: magenta ring + center dot (green = fiducial-snapped, yellow = NCC peak)."""
    vis = bgr.copy()
    for i, d in enumerate(dets):
        cx, cy = int(d["center"][0]), int(d["center"][1])
        col = (0, 255, 0) if d["source"] == "fiducial" else (0, 255, 255)
        cv2.circle(vis, (cx, cy), 60, (255, 0, 255), 4)
        cv2.circle(vis, (cx, cy), 9, col, -1)
        cv2.putText(vis, f"{i} ({d['score']:.2f})", (cx + 70, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 255), 5)
    H, W = vis.shape[:2]
    if max_width > 0 and W > max_width:
        s = max_width / float(W)
        vis = cv2.resize(vis, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return vis


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/detect_markers_v6")
def main(cfg: DictConfig):
    """Run the real-template NCC localizer over a plot, writing overlays + a JSON to v6 folders."""
    print("--- detect_markers_v6 config ---")
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

    first = cv2.imread(os.path.join(image_dir, files[0]))
    if first is None:
        print(f"ERROR: could not decode first image {files[0]}")
        return
    W0 = first.shape[1]
    work_scale = min(1.0, cfg.match_max_width / float(W0)) if cfg.match_max_width > 0 else 1.0
    templates = build_template_bank(cfg, work_scale)
    print(f"Built {len(templates)} real-template scales, full-res disk radius "
          f"{cfg.disk_radius_min}-{cfg.disk_radius_max} px; matching at {work_scale:.3f}× "
          f"({int(W0*work_scale)} px wide).")

    vis_dir = os.path.join(cfg.source_path, cfg.output_vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"v6 (real-template NCC) localizing markers in {len(files)} images from {image_dir} ...")
    print(f"Overlays → {vis_dir}/")

    per_image = {}
    counts = []
    n_fid = 0
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(image_dir, f))
        if bgr is None:
            print(f"  [skip] could not decode {f}")
            continue
        dets = detect_one(bgr, templates, work_scale, cfg)
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
    print("      MARKER LOCALIZATION v6 (real-template NCC) SUMMARY")
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
    print("NOTE: v6 is localization only — no IDs, no triangulation, data untouched.\n")


if __name__ == "__main__":
    main()
