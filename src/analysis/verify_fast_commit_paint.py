"""Offline proof that the fast bbox commit-paint in run_3d_seg.py is bit-identical to the old
full-frame paint. No GPU / no model needed — it replays both paint paths over random head alphas
(including empty heads, single-pixel heads, and multi-head overwrite order) and md5-compares the
resulting 2DSeg label maps. Reproducible: fixed seed. Run: python src/analysis/verify_fast_commit_paint.py"""
import hashlib
import numpy as np
import torch


def full_frame_paint(target, pos, head_id):
    """old behaviour: write the head id into every pixel of the full H×W frame where alpha>0.5."""
    target[pos] = head_id


def fast_bbox_paint(target, pos, head_id):
    """new behaviour: write only inside the head's 2D bounding box (bit-identical — outside the bbox
    no alpha exceeds 0.5, so nothing would be painted there anyway)."""
    rows = torch.any(pos, dim=1)
    if bool(rows.any()):
        cols = torch.any(pos, dim=0)
        ys = torch.nonzero(rows, as_tuple=False)
        xs = torch.nonzero(cols, as_tuple=False)
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        target[y0:y1, x0:x1][pos[y0:y1, x0:x1]] = head_id


def md5(t):
    return hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    H, W = 3024, 4032
    n_cams = 6
    n_heads = 40

    # one 2DSeg map per camera for each path; heads painted in the SAME order into both
    ref = [torch.zeros((H, W), dtype=torch.int) for _ in range(n_cams)]
    fast = [torch.zeros((H, W), dtype=torch.int) for _ in range(n_cams)]

    for head_id in range(1, n_heads + 1):
        for c in range(n_cams):
            pos = torch.zeros((H, W), dtype=torch.bool)
            r = torch.rand(1).item()
            if r < 0.15:
                pass  # empty head in this view (no pixels) — exercises the skip branch
            elif r < 0.25:
                # single stray pixel
                yy, xx = np.random.randint(0, H), np.random.randint(0, W)
                pos[yy, xx] = True
            else:
                # a blob (heads overlap across ids so later ids overwrite earlier — tests paint order)
                y0 = np.random.randint(0, H - 200); x0 = np.random.randint(0, W - 200)
                hh = np.random.randint(20, 200); ww = np.random.randint(20, 200)
                pos[y0:y0 + hh, x0:x0 + ww] = torch.rand(hh, ww) > 0.3  # ragged, not a clean rectangle
            full_frame_paint(ref[c], pos, head_id)
            fast_bbox_paint(fast[c], pos.clone(), head_id)

    mismatches = 0
    for c in range(n_cams):
        same = bool((ref[c] == fast[c]).all())
        h1, h2 = md5(ref[c]), md5(fast[c])
        if not same or h1 != h2:
            mismatches += 1
            print(f"  cam {c}: MISMATCH  full={h1}  fast={h2}")
        else:
            print(f"  cam {c}: OK  md5={h1}  painted_px={int((ref[c] > 0).sum())}")

    print()
    if mismatches == 0:
        print(f"PASS — fast bbox paint is BIT-IDENTICAL to full-frame paint over {n_cams} cams × {n_heads} heads.")
    else:
        print(f"FAIL — {mismatches} camera map(s) differ.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
