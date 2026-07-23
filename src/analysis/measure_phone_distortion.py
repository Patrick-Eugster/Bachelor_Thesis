"""Measure how much LENS DISTORTION the phone actually has — and whether ignoring it
(our SIMPLE_PINHOLE, which models zero distortion) can affect 3DGS.

Agisoft fit a full FULL_OPENCV distortion model (k1,k2,p1,p2,k3) to every session and stored it in the
`colmap_distorted` export (demoanlage additions). This reads those coefficients and computes, per camera,
the PIXEL DISPLACEMENT the distortion causes: how far a point moves between the ideal-pinhole position
(what SIMPLE_PINHOLE / 3DGS assume) and the true distorted position. Big peripheral displacement ⇒ our
SIMPLE_PINHOLE approximation mis-models the frame edges, which 3DGS cannot perfectly fit.

Also reports the principal-point offset Agisoft found vs the image center (our COLMAP SIMPLE_PINHOLE
keeps the principal point pinned at the center by default, so a large offset = another ignored error).

Read-only. Usage:
  python src/analysis/measure_phone_distortion.py
  python src/analysis/measure_phone_distortion.py --additions demoanlage2025_v0/demoanlage2025_v0_additions
"""
import os
import glob
import json
import argparse
import numpy as np


def parse_full_opencv(cameras_txt):
    """Read FULL_OPENCV camera lines: returns list of dicts with fx,fy,cx,cy,k1,k2,p1,p2,k3,w,h."""
    out = []
    if not os.path.isfile(cameras_txt):
        return out
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if p[1] != "FULL_OPENCV":
                continue
            w, h = int(p[2]), int(p[3])
            vals = [float(x) for x in p[4:]]
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = vals[:9]
            out.append(dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
                            k1=k1, k2=k2, p1=p1, p2=p2, k3=k3))
    return out


def displacement_stats(c, nx=80, ny=60):
    """Max / corner / center pixel displacement between ideal-pinhole and distorted projection,
    over a grid covering the image. Uses the OpenCV forward distortion model."""
    W, H = c["w"], c["h"]
    us = np.linspace(0, W - 1, nx)
    vs = np.linspace(0, H - 1, ny)
    U, V = np.meshgrid(us, vs)
    x = (U - c["cx"]) / c["fx"]
    y = (V - c["cy"]) / c["fy"]
    r2 = x * x + y * y
    radial = 1 + c["k1"] * r2 + c["k2"] * r2**2 + c["k3"] * r2**3
    xd = x * radial + 2 * c["p1"] * x * y + c["p2"] * (r2 + 2 * x * x)
    yd = y * radial + c["p1"] * (r2 + 2 * y * y) + 2 * c["p2"] * x * y
    du = c["fx"] * (xd - x)
    dv = c["fy"] * (yd - y)
    mag = np.sqrt(du * du + dv * dv)
    return dict(max_px=float(mag.max()),
                corner_px=float(mag[0, 0]),
                center_px=float(mag[ny // 2, nx // 2]),
                dcx=c["cx"] - W / 2.0, dcy=c["cy"] - H / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--additions", default="demoanlage2025_v0/demoanlage2025_v0_additions")
    ap.add_argument("--out", default="docs/analysis_results/camera_params/phone_distortion.json")
    args = ap.parse_args()

    rows = []
    for cam_txt in sorted(glob.glob(os.path.join(
            args.additions, "field_*", "*", "processed", "colmap_distorted", "sparse", "0", "cameras.txt"))):
        parts = cam_txt.split(os.sep)
        # .../field_X/DATE/processed/colmap_distorted/sparse/0/cameras.txt
        date = parts[-6]
        field = parts[-7]
        cams = parse_full_opencv(cam_txt)
        for i, c in enumerate(cams):
            s = displacement_stats(c)
            rows.append(dict(session=f"{field}/{date}", cam=i, **s,
                             k1=c["k1"], k2=c["k2"]))

    rows.sort(key=lambda r: r["max_px"], reverse=True)
    print(f"{'session':26} {'cam':3} {'max':>7} {'corner':>7} {'center':>7}   {'PP offset (dcx,dcy)':>20}")
    print("-" * 82)
    for r in rows:
        print(f"{r['session']:26} {r['cam']:<3} {r['max_px']:6.1f}p {r['corner_px']:6.1f}p "
              f"{r['center_px']:6.2f}p   ({r['dcx']:+5.0f},{r['dcy']:+5.0f}) px")

    if rows:
        mx = [r["max_px"] for r in rows]
        print(f"\nmax-displacement across {len(rows)} cameras: "
              f"min {min(mx):.0f}px  median {np.median(mx):.0f}px  max {max(mx):.0f}px")
        print("Interpretation: this is the peripheral pixel error SIMPLE_PINHOLE (zero-distortion) ignores.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"additions": args.additions, "n_cameras": len(rows), "cameras": rows}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
