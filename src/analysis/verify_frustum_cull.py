#!/usr/bin/env python3
"""Offline safety check for the segmentation_3d frustum cull (idea ① in
docs/segmentation_3d/SEGMENTATION_3D_RUNTIME.md §8).

`cull_cameras` in run_3d_seg.py drops a camera only when a head's bounding sphere lands
fully outside that camera's frustum — so flashsplat's alpha there is provably empty and
skipping the render is bit-identical to rendering it. The ONE thing that must never happen
is dropping a camera where the head IS visible (that would lose a real match / paint).

This test verifies exactly that invariant on CPU, with no model / no GPU / no rasterizer:
  (A) unit cases — a point in front-centre is KEPT; far behind / far to the side is DROPPED;
  (B) conservativeness fuzz — for many random head clouds x random cameras, the ground-truth
      "≥1 head Gaussian center projects strictly inside the image (and in front)" set must be a
      SUBSET of what cull_cameras keeps. A projection-convention bug (wrong transpose/sign) would
      drop visible cameras here and fail loudly.

Cameras are built with the SAME getWorld2View2 + getProjectionMatrix helpers the real Camera
uses (cameras.py L63-70), so the transform convention matches production exactly.

Usage:
    python src/analysis/verify_frustum_cull.py            # default 200 fuzz trials
    python src/analysis/verify_frustum_cull.py --trials 1000 --seed 1
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gaussians.utils.graphics_utils import getWorld2View2, getProjectionMatrix
from segmentation_3d.run_3d_seg import cull_cameras


class _Cam:
    """Minimal stand-in exposing only the 4 attributes cull_cameras reads, built the same
    way the real Camera does (world_view = W2V^T, full_proj = world_view @ proj^T)."""
    def __init__(self, R, T, fovx, fovy, znear=0.01, zfar=100.0):
        wv = torch.tensor(getWorld2View2(R, T).astype(np.float32)).transpose(0, 1)
        proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1)
        self.world_view_transform = wv
        self.full_proj_transform = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.FoVx = fovx
        self.FoVy = fovy


class _Gaussians:
    """Fake GaussianModel: just get_xyz + get_scaling properties over a fixed point cloud."""
    def __init__(self, xyz, scaling):
        self._xyz = xyz
        self._scaling = scaling

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scaling


def _rand_rotation(rng):
    """Random proper rotation matrix via QR of a random gaussian matrix."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))           # fix signs so it's a proper rotation
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float64)


def _center_projects_in(cam, pts):
    """Ground truth (no rasterizer): does ANY point center project strictly inside the image
    AND in front of the camera? Uses the exact same full_proj_transform convention as cull."""
    ph = torch.cat([pts, torch.ones(len(pts), 1)], dim=1)          # (P,4)
    clip = ph @ cam.full_proj_transform                            # (P,4)
    w = clip[:, 3]
    ndc = clip[:, :2] / w.unsqueeze(1)
    inside = (w > 0) & (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1)
    return bool(inside.any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200, help="random head x camera-set trials")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    fovx = fovy = np.deg2rad(60.0)

    # -------- (A) unit cases --------
    # camera at origin looking down +z (R=I, T=0 -> world_view maps world->cam identity-ish)
    cam = _Cam(np.eye(3), np.zeros(3), fovx, fovy)
    gs_center = _Gaussians(torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.01, 0.01, 0.01]]))
    gs_behind = _Gaussians(torch.tensor([[0.0, 0.0, -2.0]]), torch.tensor([[0.01, 0.01, 0.01]]))
    gs_side   = _Gaussians(torch.tensor([[50.0, 0.0, 2.0]]), torch.tensor([[0.01, 0.01, 0.01]]))
    mask1 = torch.ones(1, dtype=torch.bool)
    a_center = len(cull_cameras([cam], gs_center, mask1)) == 1   # in front centre -> kept
    a_behind = len(cull_cameras([cam], gs_behind, mask1)) == 0   # far behind      -> dropped
    a_side   = len(cull_cameras([cam], gs_side,   mask1)) == 0   # far to the side -> dropped
    print(f"(A) unit: front-centre KEPT={a_center}  behind DROPPED={a_behind}  side DROPPED={a_side}")
    unitok = a_center and a_behind and a_side

    # -------- (B) conservativeness fuzz --------
    violations = 0
    total_visible = 0
    for _ in range(args.trials):
        # a compact head cloud somewhere in a modest world volume
        n_head = int(rng.integers(20, 200))
        head_center = rng.uniform(-3, 3, size=3)
        pts = torch.tensor((head_center + 0.3 * rng.standard_normal((n_head, 3))).astype(np.float32))
        scaling = torch.full((n_head, 3), 0.02)
        gs = _Gaussians(pts, scaling)
        mask = torch.ones(n_head, dtype=torch.bool)

        # a handful of random cameras looking at random directions from random spots
        cams = []
        for _c in range(int(rng.integers(4, 12))):
            R = _rand_rotation(rng)
            T = rng.uniform(-5, 5, size=3)
            cams.append(_Cam(R, T, fovx, fovy))

        kept = set(id(c) for c in cull_cameras(cams, gs, mask))
        for c in cams:
            if _center_projects_in(c, pts):
                total_visible += 1
                if id(c) not in kept:
                    violations += 1  # a truly-visible camera was dropped -> BUG
    print(f"(B) fuzz: {args.trials} trials, {total_visible} truly-visible camera-views, "
          f"{violations} wrongly dropped")

    ok = unitok and violations == 0
    print(f"\n{'PASS ✅ — cull is conservative (never drops a visible camera)' if ok else 'FAIL ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
