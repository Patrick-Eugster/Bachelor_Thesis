"""Stage A marker localizer for phone plots. READ-ONLY on the data.

Finds the Agisoft coded ground markers (bright WHITE square plates with a black
coded disk, mounted on stakes at canopy height) in each phone image and draws an
overlay so we can eyeball whether localization works on a real green canopy.

This is ONLY localization — it answers "where is a marker in this photo" with a
sub-pixel-ish center. It does NOT read the code / assign an ID (that's Stage B,
template-matching the 6 known PDF codes) and it does NOT triangulate (Step 2).

Why localization is the risky part: the field is a dense green canopy, so we rely
on the one strong cue — the plate is much BRIGHTER and far LESS saturated than the
green wheat. We threshold on that, keep only square-ish blobs, and confirm each
blob actually contains a black disk (a blank white object would not).

This script writes NOTHING into the data except overlay PNGs under
{source_path}/marker_vis/ and one JSON under {source_path}/logs/. It never touches
images/, sparse/, etc.

Typical usage:
    python src/preprocessing/detect_markers.py field=field_A plot=20250609
    python src/preprocessing/detect_markers.py field=field_A plot=20250609 limit=5

Config: configs/preprocessing/detect_markers.yaml
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def white_plate_mask(bgr, v_min, s_max):
    """Binary mask of bright, desaturated (white/grey) pixels — the plate backing.
    The green canopy is highly saturated so it falls out; the white plate has low
    saturation and high value, so it survives. Returns a uint8 0/255 mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((v >= v_min) & (s <= s_max)).astype(np.uint8) * 255
    return mask


def clean_mask(mask, close_k, open_k):
    """Morphology to turn the speckled white-pixels into solid plate blobs.
    Close first (the black coded disk punches holes in the white plate — close fills
    them so the plate is one blob), then open to drop tiny bright noise (glare, sky)."""
    if close_k > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    if open_k > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    return mask


def dark_center(bgr, x, y, w, h, dark_v_max):
    """Find the center of the black coded disk inside a plate's bounding box.
    Returns (cx, cy, dark_fraction): the centroid of the dark pixels (the disk) and
    what fraction of the box is dark. dark_fraction is the key 'is this really a
    marker' signal — a blank white object has ~0 dark pixels inside it."""
    roi = bgr[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark = v < dark_v_max
    n_dark = int(dark.sum())
    frac = n_dark / float(w * h) if w * h > 0 else 0.0
    if n_dark == 0:
        # no disk found — fall back to the box center
        return x + w / 2.0, y + h / 2.0, 0.0
    ys, xs = np.nonzero(dark)
    cx = x + float(xs.mean())
    cy = y + float(ys.mean())
    return cx, cy, frac


def detect_one(bgr, cfg):
    """Run the full localization heuristic on one image.
    Returns a list of detections, each a dict with center, box, and the score metrics
    we filtered on (so we can inspect why something passed/failed in the JSON)."""
    H, W = bgr.shape[:2]
    img_area = float(H * W)

    mask = white_plate_mask(bgr, cfg.v_min, cfg.s_max)
    mask = clean_mask(mask, cfg.close_kernel, cfg.open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in contours:
        area = cv2.contourArea(c)
        frac = area / img_area
        # 1. size gate — plates occupy a known fraction range of the frame
        if frac < cfg.min_area_frac or frac > cfg.max_area_frac:
            continue
        # 2. squareness — fit a rotated rectangle (plates are tilted squares)
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > cfg.aspect_max:
            continue
        # squareness via CONVEX HULL fill of the rotated rect. We use the hull (not the
        # raw contour area) because the black disk reaches the plate edge and breaks the
        # white ring into a concave/fragmented contour — so raw extent is low (~0.4) even
        # for a genuine square plate, but the hull still fills the rect well (~0.85).
        hull = cv2.convexHull(c)
        extent = cv2.contourArea(hull) / (rw * rh)
        if extent < cfg.min_extent:
            continue
        # 3. must contain a black disk — the decisive marker-vs-junk test
        x, y, w, h = cv2.boundingRect(c)
        cx, cy, dark_frac = dark_center(bgr, x, y, w, h, cfg.dark_v_max)
        if dark_frac < cfg.dark_frac_min or dark_frac > cfg.dark_frac_max:
            continue
        dets.append({
            "center": [round(cx, 2), round(cy, 2)],
            "box": [int(x), int(y), int(w), int(h)],
            "area_frac": round(frac, 6),
            "aspect": round(aspect, 2),
            "extent": round(extent, 3),
            "dark_frac": round(dark_frac, 3),
        })
    return dets


def draw_overlay(bgr, dets, max_width):
    """Draw boxes + center dots + a candidate index on a copy of the image.
    Downscale the saved overlay so 119 PNGs don't eat disk — marks scale with it."""
    vis = bgr.copy()
    for i, d in enumerate(dets):
        x, y, w, h = d["box"]
        cx, cy = d["center"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1)
        cv2.putText(vis, str(i), (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    H, W = vis.shape[:2]
    if max_width > 0 and W > max_width:
        scale = max_width / float(W)
        vis = cv2.resize(vis, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    return vis


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/detect_markers")
def main(cfg: DictConfig):
    """Localize markers in every image, write overlay PNGs + a detections JSON.
    Stage A: no IDs, no triangulation — just 'did we find the plates'. Eyeball the
    overlays in marker_vis/ before trusting any downstream geometry."""
    print("--- detect_markers config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------")
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
    print(f"Localizing markers in {len(files)} images from {image_dir} ...")
    print(f"Overlays → {vis_dir}/")

    per_image = {}
    counts = []
    for i, f in enumerate(files):
        path = os.path.join(image_dir, f)
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  [skip] could not decode {f}")
            continue
        dets = detect_one(bgr, cfg)
        per_image[f] = dets
        counts.append(len(dets))
        vis = draw_overlay(bgr, dets, cfg.overlay_max_width)
        cv2.imwrite(os.path.join(vis_dir, f), vis)
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {f:<32} → {len(dets)} candidate(s)")

    counts = np.array(counts) if counts else np.array([0])
    # how many images hit the expected marker count (rough health signal)
    n_at_expected = int(np.sum(counts == cfg.expected_markers))
    n_zero = int(np.sum(counts == 0))

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("      MARKER LOCALIZATION (Stage A) SUMMARY")
    print("=" * 60)
    print(f"{'Plot:':<30} {cfg.field}/{cfg.plot}")
    print(f"{'Images processed:':<30} {len(per_image)}")
    print(f"{'Expected markers/image:':<30} {cfg.expected_markers} (not all visible per view)")
    print("-" * 60)
    print(f"{'candidates/image  min/max:':<30} {counts.min()} / {counts.max()}")
    print(f"{'candidates/image  mean:':<30} {counts.mean():.2f}")
    print(f"{'candidates/image  median:':<30} {int(np.median(counts))}")
    print(f"{'images with 0 candidates:':<30} {n_zero}")
    print(f"{'images with exactly {0}:'.format(cfg.expected_markers):<30} {n_at_expected}")
    print(f"{'total candidates:':<30} {int(counts.sum())}")
    print("-" * 60)
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"{'TOTAL TIME:':<30} {minutes}m {seconds}s")
    print("=" * 60 + "\n")
    print("NEXT: open a few overlays in marker_vis/ and check —")
    print("  are all visible plates boxed? any false positives in the canopy?")
    print("  then we tune thresholds, and move to Stage B (IDs via template-match).\n")

    report = {
        "field": cfg.field,
        "plot": cfg.plot,
        "image_subdir": cfg.image_subdir,
        "n_images": len(per_image),
        "expected_markers": cfg.expected_markers,
        "params": {
            "v_min": cfg.v_min, "s_max": cfg.s_max, "dark_v_max": cfg.dark_v_max,
            "close_kernel": cfg.close_kernel, "open_kernel": cfg.open_kernel,
            "min_area_frac": cfg.min_area_frac, "max_area_frac": cfg.max_area_frac,
            "min_extent": cfg.min_extent, "aspect_max": cfg.aspect_max,
            "dark_frac_min": cfg.dark_frac_min, "dark_frac_max": cfg.dark_frac_max,
        },
        "counts": {
            "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(counts.mean()), "median": float(np.median(counts)),
            "n_zero": n_zero, "n_at_expected": n_at_expected,
            "total": int(counts.sum()),
        },
        "per_image": per_image,
        "elapsed_s": elapsed,
    }
    out_path = os.path.join(cfg.source_path, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Detections JSON written to {out_path}\n")
    print("NOTE: Stage A is localization only — no IDs, no triangulation, data untouched.\n")


if __name__ == "__main__":
    main()
