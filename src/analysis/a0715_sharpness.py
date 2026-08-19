"""Compute whole-image sharpness (variance-of-Laplacian ratio) locally for the A/0715
COLMAP-vs-Agisoft densification runs, so the single-session thesis table (tab:recon-phone-a0715)
can carry a sharpness column. Pure CPU, no VRAM: reads the already-rendered test PNGs.

Replicates src/reconstruction/metrics.py::laplacian_var exactly (same kernel, grayscale*255,
torch .var()) so the numbers match the values metrics.py already wrote for the other runs.
We validate against a run whose sharpness is stored in results.json before trusting the rest.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_LAP_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def laplacian_var(img):
    """Same as metrics.py: img is [1,3,H,W] in [0,1], grayscale*255, variance of its Laplacian."""
    gray = img.mean(1, keepdim=True) * 255.0
    lap = F.conv2d(gray, _LAP_KERNEL, padding=1)
    return lap.var().item()


def load(png):
    """Load a PNG as a [1,3,H,W] float tensor in [0,1]."""
    arr = np.array(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def run_dir_sharpness(test_iter_dir):
    """Mean render + gt Laplacian variance over the matched test PNGs, and their ratio."""
    rdir, gdir = test_iter_dir / "renders", test_iter_dir / "gt"
    names = sorted(p.name for p in rdir.glob("*.png"))
    sr, sg = [], []
    for n in names:
        sr.append(laplacian_var(load(rdir / n)))
        sg.append(laplacian_var(load(gdir / n)))
    mr, mg = float(np.mean(sr)), float(np.mean(sg))
    return mr, mg, (mr / mg if mg > 0 else 0.0), len(names)


BASE = Path("results/reconstruction/phone/field_A/20250715")
RUNS = [
    ("COLMAP opencv 15k default", BASE / "opencv/vanilla_3dgs/baseline/test/ours_15000"),
    ("COLMAP opencv 15k AbsGS",   BASE / "opencv/vanilla_3dgs/absgrad/test/ours_15000"),
    ("COLMAP opencv 30k AbsGS",   BASE / "opencv/vanilla_3dgs/dense17k/test/ours_30000"),
    ("Agisoft 15k default",       BASE / "agisoft/vanilla_3dgs/agisoft_bench/test/ours_15000"),
    ("Agisoft 15k AbsGS",         BASE / "agisoft/vanilla_3dgs/agisoft_absgrad/test/ours_15000"),
    ("Agisoft 30k AbsGS",         BASE / "agisoft/vanilla_3dgs/agisoft_dense17k/test/ours_30000"),
]

if __name__ == "__main__":
    # validation: opencv baseline stored render=93.383 gt=624.819 ratio=0.1495 in its results.json
    mr, mg, ratio, n = run_dir_sharpness(RUNS[0][1])
    print(f"VALIDATION opencv baseline: render={mr:.3f} gt={mg:.3f} ratio={ratio:.4f} (n={n})")
    print("  expected from results.json: render=93.383 gt=624.819 ratio=0.1495\n")
    print(f"{'run':<28}{'sharp_render':>13}{'sharp_gt':>10}{'ratio':>8}")
    for label, d in RUNS:
        mr, mg, ratio, n = run_dir_sharpness(d)
        print(f"{label:<28}{mr:>13.2f}{mg:>10.2f}{ratio:>8.3f}")
