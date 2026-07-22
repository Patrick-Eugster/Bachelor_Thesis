"""Draw Agisoft's coded markers onto Agisoft's own images, for all phone sessions — pure Agisoft data.

IMPORTANT (coordinate space): Agisoft's marker_projections.csv 2D detections live in the DISTORTED
reconstruction `.../processed/colmap_distorted/` (FULL_OPENCV, 4032x3024, real distortion + off-centre
principal point) — NOT the undistorted `input_plots/.../agisoft/sparse/` (SIMPLE_PINHOLE, 3850x2878).
Using the undistorted model put markers ~25 px off; using colmap_distorted (via pycolmap, which applies
the FULL_OPENCV distortion) reprojects to ~2-4 px. So everything here uses colmap_distorted + its images.

Modes:
  --mode raw      draw ONLY Agisoft's raw 2D detections (1-2 markers/image) — pure, no computation.
  --mode project  (default) triangulate each marker ONCE from Agisoft's own 2D detections + Agisoft poses
                  (undistort each detection via pycolmap `cam_from_img`, DLT), then project all 6 into
                  every rendered image via `project_point` (distortion-correct). Lets you see which of the
                  6 fall in each frame. Raw detections are also drawn (small crosses) as a cross-check.
Nothing from OUR reconstruction is used — only Agisoft's poses + detections. Renders N evenly-spaced
images per session (the 3D is still triangulated from ALL detections). TARGET_TO_CODE labels by our IDs.

Usage:
  python src/analysis/overlay_agisoft_markers_all.py                          # all sessions, project, 6/img
  python src/analysis/overlay_agisoft_markers_all.py --mode raw
  python src/analysis/overlay_agisoft_markers_all.py --field field_D --plot 20250706 --sample 12
"""

import os
import csv
import argparse

import numpy as np
import cv2
import pycolmap as pc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_TO_CODE = {1: 113, 2: 105, 3: 89, 4: 101, 5: 85, 6: 77}
COLORS = {77: (0, 0, 255), 85: (0, 165, 255), 89: (0, 255, 255),
          101: (0, 255, 0), 105: (255, 128, 0), 113: (255, 0, 255)}


def base_name(full):
    """'IMG_20250706_123048_23.jpg' -> 'IMG_20250706_123048' (strip trailing _<seq> + extension)."""
    stem = os.path.splitext(full)[0]
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem


def load_detections(csv_path):
    """{base_camera: {code: (x,y)}} from Agisoft's marker_projections.csv (distorted-space pixels)."""
    out = {}
    for r in csv.DictReader(open(csv_path)):
        tnum = int(r["Marker"].replace("target", "").strip())
        if tnum not in TARGET_TO_CODE:
            continue
        out.setdefault(r["Camera"], {})[TARGET_TO_CODE[tnum]] = (float(r["X"]), float(r["Y"]))
    return out


def pose_reproj(im, rec, max_pts=80):
    """Median scene-point reprojection error (px) for one image = its Agisoft POSE reliability.
    Low = trustworthy pose; high = unreliable (weak reconstruction, e.g. early-season low texture).
    Capped at max_pts points for speed."""
    errs = []
    for p2 in im.points2D:
        if p2.has_point3D():
            uv = im.project_point(rec.point3D(p2.point3D_id).xyz)
            if uv is not None:
                errs.append(np.hypot(uv[0] - p2.xy[0], uv[1] - p2.xy[1]))
            if len(errs) >= max_pts:
                break
    return float(np.median(errs)) if errs else None


def triangulate(views):
    """DLT from [(R, t, normalized_undistorted_xy), ...] (P has no K — coords already undistorted)."""
    A = []
    for R, t, n in views:
        P = np.hstack([R, t.reshape(3, 1)])
        A.append(n[0] * P[2] - P[0])
        A.append(n[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def process_session(field, plot, mode, downscale, sample, pose_thresh, reliable_only):
    proc = os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions",
                        field, plot, "processed")
    spdir = os.path.join(proc, "colmap_distorted", "sparse", "0")
    imdir = os.path.join(proc, "colmap_distorted", "images")
    csv_path = os.path.join(proc, "marker_projections.csv")
    if not (os.path.isdir(spdir) and os.path.isdir(imdir) and os.path.exists(csv_path)):
        return 0

    rec = pc.Reconstruction(spdir)
    imgs_by_base = {base_name(im.name): im for im in rec.images.values()}
    det = load_detections(csv_path)

    # per-marker 3D from Agisoft's own detections+poses (project mode) + marker-triangulation quality
    marker3d = {}
    marker_reproj = []
    if mode == "project":
        by_code = {}
        for cam_base, dd in det.items():
            im = imgs_by_base.get(cam_base)
            if im is None:
                continue
            camera = rec.camera(im.camera_id)
            T = im.cam_from_world()
            R, t = T.rotation.matrix(), np.array(T.translation)
            for code, xy in dd.items():
                n = np.array(camera.cam_from_img(np.array(xy)))     # undistort -> normalized ray
                by_code.setdefault(code, []).append((R, t, n, im, np.array(xy)))
        for code, views in by_code.items():
            if len(views) >= 2:
                X = triangulate([(R, t, n) for R, t, n, _, _ in views])
                marker3d[code] = X
                for _, _, _, im, xy in views:
                    uv = im.project_point(X)
                    if uv is not None:
                        marker_reproj.append(np.hypot(uv[0] - xy[0], uv[1] - xy[1]))

    suffix = mode + ("_reliable" if reliable_only else "")
    out_dir = os.path.join(REPO, "input_plots", "phone", field, plot, "agisoft",
                           "marker_vis_agisoft_" + suffix)
    os.makedirs(out_dir, exist_ok=True)
    all_imgs = sorted(rec.images.values(), key=lambda im: im.name)
    pr_cache = {}
    if reliable_only:                                   # keep only frames with a trustworthy Agisoft pose
        cand = []
        for im in all_imgs:
            pr = pose_reproj(im, rec)
            pr_cache[im.name] = pr
            if pr is not None and pr <= pose_thresh:
                cand.append(im)
    else:
        cand = all_imgs
    render = cand
    if sample and len(render) > sample:
        idx = np.linspace(0, len(render) - 1, sample).round().astype(int)
        render = [render[i] for i in sorted(set(idx))]

    n_written = 0
    pose_reprojs = []
    for im in render:
        path = os.path.join(imdir, im.name)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        in_frame = []

        if mode == "project":
            for code, X in marker3d.items():
                uv = im.project_point(X)                 # distortion-correct projection, or None
                if uv is None:
                    continue
                u, v = float(uv[0]), float(uv[1])
                if 0 <= u <= W and 0 <= v <= H:
                    in_frame.append(code)
                col = COLORS.get(code, (255, 255, 255))
                p = (int(np.clip(u, 4, W - 4)), int(np.clip(v, 4, H - 4)))
                cv2.circle(img, p, 30, col, -1)
                cv2.circle(img, p, 30, (0, 0, 0), 3)
                cv2.putText(img, str(code), (p[0] + 34, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)

        for code, (x, y) in det.get(base_name(im.name), {}).items():
            col = COLORS.get(code, (255, 255, 255))
            cv2.drawMarker(img, (int(x), int(y)), col, cv2.MARKER_TILTED_CROSS, 46, 6)
            if mode == "raw":
                in_frame.append(code)
                cv2.putText(img, str(code), (int(x) + 28, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)

        pr = pr_cache.get(im.name)
        if pr is None:
            pr = pose_reproj(im, rec)
        if pr is not None:
            pose_reprojs.append(pr)
        reliable = pr is not None and pr <= pose_thresh
        posetag = "" if mode == "raw" else f"   pose={pr:.0f}px {'OK' if reliable else 'UNRELIABLE'}"
        cv2.rectangle(img, (0, 0), (W, 96), (30, 30, 30), -1)
        tag = "detected" if mode == "raw" else "in-frame (projected)"
        title = f"{im.name}   Agisoft markers {tag}: {sorted(set(in_frame))} = {len(set(in_frame))}/6{posetag}"
        col_title = (255, 255, 255) if (mode == "raw" or reliable) else (80, 80, 255)
        cv2.putText(img, title, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.15, col_title, 3)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(im.name)[0] + f"_{mode}.jpg"),
                    cv2.resize(img, (W // downscale, H // downscale)))
        n_written += 1
    mreproj = float(np.median(marker_reproj)) if marker_reproj else None
    preproj = float(np.median(pose_reprojs)) if pose_reprojs else None
    n_reliable = len(cand) if reliable_only else None
    return n_written, out_dir, sorted(marker3d), mreproj, preproj, n_reliable, len(all_imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--mode", choices=["raw", "project"], default="project")
    ap.add_argument("--downscale", type=int, default=3)
    ap.add_argument("--sample", type=int, default=6, help="render N evenly-spaced images/session (0 = all)")
    ap.add_argument("--reliable-only", action="store_true",
                    help="project mode: skip frames whose Agisoft pose is unreliable (scene reproj > --pose-thresh)")
    ap.add_argument("--pose-thresh", type=float, default=12.0, help="px; pose reliability cutoff (scene reproj)")
    args = ap.parse_args()

    root = os.path.join(REPO, "demoanlage2025_v0", "demoanlage2025_v0_additions")
    sessions = []
    for fld in ([args.field] if args.field else sorted(os.listdir(root)) if os.path.isdir(root) else []):
        fdir = os.path.join(root, fld)
        if not os.path.isdir(fdir):
            continue
        for plt in ([args.plot] if args.plot else sorted(os.listdir(fdir))):
            if os.path.isdir(os.path.join(fdir, plt, "processed", "colmap_distorted", "sparse", "0")):
                sessions.append((fld, plt))

    rel = " (reliable-only)" if args.reliable_only else ""
    print(f"mode={args.mode}{rel}; sample={args.sample}/session; pose_thresh={args.pose_thresh}px; "
          f"{len(sessions)} sessions\n")
    print(f"  {'session':<26} {'imgs':>10}  {'marker_reproj':>13}  {'pose_reproj':>11}")
    for field, plot in sessions:
        res = process_session(field, plot, args.mode, args.downscale, args.sample,
                              args.pose_thresh, args.reliable_only)
        if res == 0:
            print(f"  {field+'/'+plot:<26} skipped (missing colmap_distorted/images/csv)")
            continue
        n, out_dir, m3d, mreproj, preproj, n_reliable, n_total = res
        imgs = f"{n_reliable}/{n_total} rel" if n_reliable is not None else f"{n} rendered"
        mr = f"{mreproj:.1f}px" if mreproj is not None else "-"
        prs = f"{preproj:.0f}px" if preproj is not None else "-"
        flag = "  <-- POOR SESSION" if (preproj is not None and preproj > args.pose_thresh) else ""
        print(f"  {field+'/'+plot:<26} {imgs:>10}  {mr:>13}  {prs:>11}{flag}")


if __name__ == "__main__":
    main()
