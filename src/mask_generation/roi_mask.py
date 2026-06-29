"""
roi_mask.py — restrict mask generation to a region of interest (ROI) per image.

Phone captures have bad image corners (lens distortion, blur, and wheat heads that
belong to neighbouring plots), so YOLO/SAHI/SAM waste effort there and pick up junk.
This module builds a per-image ROI from the ground markers and (1) greys-out everything
outside it BEFORE inference, and (2) post-filters the resulting boxes, so the models only
ever work on the actual plot.

HOW THE ROI IS BUILT (source="markers", the phone case):
We already triangulated the 6 coded ground markers to 3D (logs/marker_points3d.json,
in the sparse/0 reconstruction frame). For each image we project those 3D points into
that camera using its sparse/0 pose + intrinsics, take the convex hull of the projected
points → a polygon = the physical plot boundary as seen from that view. Because we
project the *3D* markers (not the per-image 2D detections), every image gets a polygon
even if a marker wasn't directly detected in it, and the polygon is geometrically
consistent across views.

SOFT BORDER (not a hard cut):
The hull joins marker CENTRES, so heads sitting on the boundary would be cut in half.
So the mask is the polygon GROWN OUTWARD by roi.buffer_px pixels — boundary heads stay
fully visible to YOLO/SAM. Then roi.filter_boxes drops boxes that fall outside the plot:
  filter_mode="overlap" (default, lenient) → drop a box only if its bounding box is
      COMPLETELY outside the un-buffered (true) polygon; any box that overlaps the polygon
      at all is kept (even if its centre is outside). Maximises recall on boundary heads.
  filter_mode="center" (strict) → drop a box whose CENTRE is outside the true polygon.
roi.filter_tol_px softens the boundary (a box within tol px of the polygon still counts).

FALLBACK: if fewer than roi.min_markers markers can be projected (e.g. FIP, which has no
phone-coded markers, or a failed session), we fall back to roi.fallback:
  none   → no masking and no filtering (byte-identical to not using ROI) — the safe default
  circle → central circle (targets radial corner distortion, marker-free)
  square → central square inset

This is OFF by default (roi.enabled=false) → byte-identical to the old pipeline. It's a
phone-only feature: turn it on with roi.enabled=true on phone runs; on FIP it auto-falls
back (no phone markers present).

FRAME NOTE: mask generation runs on the undistorted images/ folder, and sparse/0 +
marker_points3d.json are in that same undistorted frame, so the projection lands directly
in images/ pixel space — no undistortion of the projected points needed.
"""

import os
import json
import threading

import numpy as np
import cv2

# cache one parsed plot (per-image base polygons) so the pipelined loaders don't re-read
# sparse/0 for every image. keyed by plot dir.
_PLOT_CACHE = {}
_CACHE_LOCK = threading.Lock()


# =====================================================================
# --- COLMAP sparse/0 parsing ---
# =====================================================================

def _quat_to_rotmat(qw, qx, qy, qz):
    """COLMAP world-to-camera rotation from a (qw,qx,qy,qz) quaternion (normalized first)."""
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def _parse_cameras(path):
    """Read cameras.txt → {cam_id: (model, w, h, [params])}. Handles SIMPLE_PINHOLE + PINHOLE."""
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            cam_id = int(t[0])
            model = t[1]
            w, h = int(t[2]), int(t[3])
            params = [float(x) for x in t[4:]]
            cams[cam_id] = (model, w, h, params)
    return cams


def _parse_images(path):
    """Read images.txt → {name: (R, t, cam_id)}. Only the first (pose) line per image is used."""
    imgs = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue
        t = line.split()
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        qw, qx, qy, qz = map(float, t[1:5])
        tx, ty, tz = map(float, t[5:8])
        cam_id = int(t[8])
        name = t[9]
        R = _quat_to_rotmat(qw, qx, qy, qz)
        tvec = np.array([tx, ty, tz])
        imgs[name] = (R, tvec, cam_id)
        i += 2  # skip the POINTS2D line
    return imgs


def _project(xyz, R, tvec, model, params):
    """Project one 3D world point into pixels. Returns (u,v) or None if behind the camera."""
    Xc = R @ np.asarray(xyz) + tvec
    if Xc[2] <= 1e-6:
        return None  # behind the camera
    x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
    if model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        # SIMPLE_PINHOLE (and any undistorted single-focal model): f, cx, cy
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    return (fx * x + cx, fy * y + cy)


# =====================================================================
# --- per-plot ROI build (cached) ---
# =====================================================================

def _build_plot_polys(plot_dir, min_markers):
    """Project the 3D markers into every image of one plot → {name: base_poly(Nx2) or None}.
    None means too few markers projected for that image (caller applies the fallback).
    The stored polygon is the raw marker hull (NO buffer) — the buffer is added at mask time
    so the same base polygon also drives the box centre-filter."""
    sparse_dir = os.path.join(plot_dir, "sparse", "0")
    cam_path = os.path.join(sparse_dir, "cameras.txt")
    img_path = os.path.join(sparse_dir, "images.txt")
    mk_path = os.path.join(plot_dir, "logs", "marker_points3d.json")

    # missing any input → no markers anywhere → every image falls back
    if not (os.path.isfile(cam_path) and os.path.isfile(img_path) and os.path.isfile(mk_path)):
        return {}

    cams = _parse_cameras(cam_path)
    imgs = _parse_images(img_path)
    with open(mk_path) as f:
        mk = json.load(f)
    marker_xyz = [v["xyz"] for v in mk.get("points3d", {}).values()]

    polys = {}
    for name, (R, tvec, cam_id) in imgs.items():
        model, w, h, params = cams[cam_id]
        pts = []
        for xyz in marker_xyz:
            uv = _project(xyz, R, tvec, model, params)
            if uv is not None:
                pts.append(uv)
        if len(pts) < min_markers:
            polys[name] = None  # fallback for this image
            continue
        hull = cv2.convexHull(np.array(pts, dtype=np.float32)).reshape(-1, 2)
        polys[name] = np.round(hull).astype(np.int32)
    return polys


def _get_plot_cache(plot_dir, roi):
    """Lazily parse+cache one plot's ROI base polygons. Thread-safe for the pipelined loaders."""
    with _CACHE_LOCK:
        if plot_dir not in _PLOT_CACHE:
            _PLOT_CACHE[plot_dir] = _build_plot_polys(plot_dir, int(roi.get("min_markers", 3)))
        return _PLOT_CACHE[plot_dir]


def _resolve_buffer_px(roi, w, h):
    """How many pixels to grow the mask polygon. buffer_frac (fraction of the image short side)
    scales with resolution and wins; buffer_px is the absolute fallback when buffer_frac<=0."""
    frac = float(roi.get("buffer_frac", 0.0))
    if frac > 0:
        return int(round(frac * min(w, h)))
    return int(roi.get("buffer_px", 0))


def _roi_keep_region(poly, w, h, buffer_px):
    """Boolean HxW mask of the kept region = polygon grown outward by buffer_px.
    Uses distanceTransform (distance-to-polygon <= buffer) instead of a morphological dilate:
    a big buffer needs a (2*buffer+1)^2 kernel which is O(buffer^2) and gets very slow (~6s/img at
    buffer=151), while distanceTransform is ~constant ~0.08s/img for the same result."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    if buffer_px <= 0:
        return mask == 1
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)  # px distance from outside → polygon
    return (mask == 1) | (dist <= buffer_px)


def _fallback_polygon(kind, w, h, roi):
    """Marker-free ROI as a convex polygon (so mask + centre-filter share one code path)."""
    if kind == "circle":
        frac = float(roi.get("fallback_circle_frac", 0.9))
        r = int(frac * min(w, h) / 2)
        # 72-gon approximation of the circle — convex, works with fillConvexPoly + pointPolygonTest
        return cv2.ellipse2Poly((w // 2, h // 2), (r, r), 0, 0, 360, 5).astype(np.int32)
    if kind == "square":
        frac = float(roi.get("fallback_square_frac", 0.9))
        mw, mh = (1 - frac) * w / 2, (1 - frac) * h / 2
        return np.array([[mw, mh], [w - mw, mh], [w - mw, h - mh], [mw, h - mh]], dtype=np.int32)
    return None  # "none" → no ROI


def _base_polygon(img_path, cfg, w, h):
    """Resolve the (un-buffered) ROI polygon for one image: marker hull, else fallback shape,
    else None (= ROI disabled / fallback none → no masking, no filtering)."""
    roi = cfg.get("roi", None)
    if roi is None or not roi.get("enabled", False):
        return None, roi
    poly = None
    if roi.get("source", "markers") == "markers":
        plot_dir = os.path.dirname(os.path.dirname(os.path.abspath(img_path)))
        poly = _get_plot_cache(plot_dir, roi).get(os.path.basename(img_path), None)
    if poly is None:
        poly = _fallback_polygon(roi.get("fallback", "none"), w, h, roi)
    return poly, roi


# =====================================================================
# --- public entry points ---
# =====================================================================

def apply_roi(img, img_path, cfg):
    """Grey-out everything outside the per-image ROI (the polygon GROWN by roi.buffer_px so
    boundary heads aren't cut). Returns a masked copy; no-op when ROI is disabled / fallback none.
    img = HxWx3 uint8 (RGB or BGR — the grey fill is channel-symmetric so order doesn't matter)."""
    h, w = img.shape[:2]
    poly, roi = _base_polygon(img_path, cfg, w, h)
    if poly is None:
        return img

    # soft border: grow the kept region outward so heads straddling the marker line survive.
    # buffer_frac (fraction of the image short side) scales with resolution and takes precedence;
    # buffer_px is an absolute override used only when buffer_frac <= 0.
    buffer_px = _resolve_buffer_px(roi, w, h)
    keep = _roi_keep_region(poly, w, h, buffer_px)

    fill = tuple(int(c) for c in roi.get("fill", [114, 114, 114]))
    out = img.copy()
    out[~keep] = fill
    return out


def roi_keep_mask(boxes_xyxy, img_path, cfg, img_w=None, img_h=None):
    """Boolean keep-mask for boxes against the un-buffered (true) marker polygon.
      filter_mode="overlap" (default): keep unless the box is COMPLETELY outside the polygon
          (any intersection → keep, even with the centre outside) — lenient, recall-friendly.
      filter_mode="center": keep only if the box CENTRE is inside the polygon.
    roi.filter_tol_px softens the boundary in both modes.
    Returns all-True when ROI disabled, filter_boxes off, or no polygon for this image.
    boxes_xyxy = (N,4) array-like [x1,y1,x2,y2]; img_w/img_h needed only for fallback shapes."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    if n == 0:
        return np.ones(0, dtype=bool)

    roi = cfg.get("roi", None)
    if roi is None or not roi.get("enabled", False) or not roi.get("filter_boxes", True):
        return np.ones(n, dtype=bool)

    # need image size for fallback shapes; for marker polygons it's unused
    w = int(img_w) if img_w else 0
    h = int(img_h) if img_h else 0
    poly, _ = _base_polygon(img_path, cfg, w, h)
    if poly is None:
        return np.ones(n, dtype=bool)

    tol = float(roi.get("filter_tol_px", 0.0))
    mode = roi.get("filter_mode", "overlap")
    poly_f = poly.astype(np.float32)
    keep = np.empty(n, dtype=bool)

    if mode == "center":
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        for i in range(n):
            # pointPolygonTest: + inside, 0 on edge, − outside (signed distance in px)
            keep[i] = cv2.pointPolygonTest(poly_f, (float(cx[i]), float(cy[i])), True) >= -tol
        return keep

    # "overlap" (default): keep if the box (grown by tol) intersects the polygon at all
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = x1 - tol, y1 - tol, x2 + tol, y2 + tol
        box_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        area, _ = cv2.intersectConvexConvex(poly_f, box_poly)
        keep[i] = area > 0.0  # area 0 → bounding box completely outside the plot → drop
    return keep


# =====================================================================
# --- debug CLI: dump ROI overlays for one plot to eyeball before a real run ---
# =====================================================================

if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Dump ROI overlays for one plot (visual check).")
    ap.add_argument("plot_dir", help="e.g. input_plots/phone/field_A/20250715")
    ap.add_argument("--out", default="/tmp/roi_overlay", help="output folder for overlays")
    ap.add_argument("--min_markers", type=int, default=3)
    ap.add_argument("--buffer_px", type=int, default=0, help="absolute buffer (used if --buffer_frac<=0)")
    ap.add_argument("--buffer_frac", type=float, default=0.045, help="buffer as fraction of image short side")
    ap.add_argument("--limit", type=int, default=8, help="how many images to draw")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    polys = _build_plot_polys(args.plot_dir, args.min_markers)
    img_dir = os.path.join(args.plot_dir, "images")
    files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    n_with, n_without = 0, 0
    for fp in files[: args.limit]:
        name = os.path.basename(fp)
        poly = polys.get(name)
        img = cv2.imread(fp)
        if poly is None:
            n_without += 1
            cv2.imwrite(os.path.join(args.out, f"NOPOLY_{name}"), img)
            continue
        n_with += 1
        h, w = img.shape[:2]
        buf = int(round(args.buffer_frac * min(w, h))) if args.buffer_frac > 0 else args.buffer_px
        # buffered mask (what YOLO/SAM actually see) shown as the dimmed-out region
        mask = _roi_keep_region(poly, w, h, buf).astype(np.uint8)
        vis = img.copy()
        vis[mask == 0] = (vis[mask == 0] * 0.35).astype(np.uint8)  # darken outside the buffered ROI
        cv2.polylines(vis, [poly], True, (0, 255, 0), 4)            # green = true marker hull (filter line)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 255), 3)          # orange = buffered mask edge
        cv2.imwrite(os.path.join(args.out, f"roi_{name}"), vis)
    print(f"wrote overlays for {min(len(files), args.limit)} images → {args.out}")
    print(f"images with marker polygon: {n_with}, without (would fall back): {n_without}, total in plot: {len(polys)}")
