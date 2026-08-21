"""Checks whether the FIP default z<z_mean ground cull drops real wheat heads on plot_461. The cull keeps
only Gaussians above the mean scene height; from the overhead GT camera the visible surface of a head is
its TOP (highest Gaussians, nearest the camera), so a head survives iff its top is above z_mean and is
fully dropped iff its whole extent is below z_mean. We project every Gaussian into the labeled GT camera
using the COLMAP sparse pose (unambiguous, incl. cx/cy), bin them into the hand-labeled GT head boxes, and
for each box compare the MAX Gaussian height inside it to z_mean. Boxes whose max height < z_mean are heads
the cull removes entirely. Read-only; nothing is modified. Run from repo root."""
import os
import numpy as np
from plyfile import PlyData

PLOT = "plot_461"
STEM = "FPWW036_SR0461_FIP2_cam_12"
PLY = f"results/reconstruction/fip/{PLOT}/vanilla_3dgs/fipseg15k_pp/point_cloud/iteration_15000/point_cloud.ply"
SPARSE = f"input_plots/fip/{PLOT}/sparse/0"
GT_TXT = f"input_plots/fip/{PLOT}/manual_label/{STEM}.txt"


def qvec2rot(q):
    """COLMAP quaternion (w,x,y,z) -> 3x3 rotation (world->camera)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x*x+y*y)],
    ])


def load_gt_cam():
    """Reads the GT camera's qvec,tvec + intrinsics (fx,fy,cx,cy,W,H) from COLMAP sparse text."""
    cam_id = qw = None
    for line in open(f"{SPARSE}/images.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 10 and p[9].rsplit(".", 1)[0] == STEM:
            qw = [float(x) for x in p[1:5]]
            tv = [float(x) for x in p[5:8]]
            cam_id = p[8]
            break
    assert qw is not None, f"{STEM} not in images.txt"
    intr = None
    for line in open(f"{SPARSE}/cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if p[0] == cam_id:
            model, W, H = p[1], int(p[2]), int(p[3])
            params = [float(x) for x in p[4:]]
            if model == "SIMPLE_PINHOLE":
                fx = fy = params[0]; cx, cy = params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            intr = (fx, fy, cx, cy, W, H)
            break
    return qvec2rot(qw), np.array(tv), intr


def main():
    xyz = np.stack([np.asarray(PlyData.read(PLY)["vertex"][k]) for k in ("x", "y", "z")], 1).astype(np.float64)
    z = xyz[:, 2]
    z_mean = z.mean()
    cull = z < z_mean
    print(f"Gaussians: {len(xyz)}   z_mean={z_mean:.3f}")
    print(f"z<z_mean cull: {cull.sum()}/{len(xyz)} = {100*cull.mean():.1f}% removed\n")

    R, t, (fx, fy, cx, cy, W, H) = load_gt_cam()
    Xc = (R @ xyz.T).T + t                       # world -> camera
    front = Xc[:, 2] > 1e-6
    u = fx * Xc[:, 0] / Xc[:, 2] + cx
    v = fy * Xc[:, 1] / Xc[:, 2] + cy
    inframe = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    print(f"projected in front & in-frame: {inframe.sum()}/{len(xyz)}")

    # GT head boxes (yolo normalized) -> pixel xyxy
    boxes = []
    for line in open(GT_TXT):
        p = line.split()
        if len(p) >= 5:
            _, bx, by, bw, bh = (float(x) for x in p[:5])
            boxes.append([(bx-bw/2)*W, (by-bh/2)*H, (bx+bw/2)*W, (by+bh/2)*H])
    boxes = np.array(boxes)
    print(f"GT head boxes: {len(boxes)}\n")

    dropped = 0
    partial = 0
    kept_fracs = []
    n_with_g = 0
    for (x1, y1, x2, y2) in boxes:
        inbox = inframe & (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
        if inbox.sum() < 3:
            continue                               # too few gaussians to judge
        n_with_g += 1
        zbox = z[inbox]
        kept_frac = (zbox >= z_mean).mean()
        kept_fracs.append(kept_frac)
        if zbox.max() < z_mean:
            dropped += 1                            # entire head below mean -> fully culled
        elif kept_frac < 0.5:
            partial += 1                            # top survives but most of the head is culled
    print(f"heads with enough gaussians to judge: {n_with_g}/{len(boxes)}")
    print(f"FULLY DROPPED (max height < z_mean): {dropped}")
    print(f"heavily culled (top kept but <50% of head above z_mean): {partial}")
    if kept_fracs:
        kf = np.array(kept_fracs)
        print(f"per-head kept-fraction: mean {kf.mean():.2f}, median {np.median(kf):.2f}, "
              f"min {kf.min():.2f}, %heads>80%kept {100*(kf>0.8).mean():.0f}%")


if __name__ == "__main__":
    main()
