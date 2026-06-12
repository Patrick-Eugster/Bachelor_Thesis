"""
Diagnostic: compare two bboxes/ output folders from the mask-generation stage.

Use it to check that a refactor or a config change did not alter the detector's
output — e.g. after the run_mask_generation.py orchestrator refactor, run YOLO+SAM
twice (old entry vs new entry) into two experiment folders and compare their
bboxes/. Each *.pt holds the per-image good boxes [N,4] in original-image coords
(empty tensor when an image has no heads).

NOTE on GPU non-determinism: two runs of the *same* code can differ by a handful
of boxes, because a detection sitting right on the conf threshold can flip in/out
between runs. So a few single-box count diffs are normal — only a large gap or a
big coordinate shift signals a real change. Read-only; never modifies anything.

Run:
    python src/analysis/compare_bboxes.py <dirA> <dirB>
    python src/analysis/compare_bboxes.py <dirA> <dirB> --atol 0.01
where <dirA>/<dirB> are two bboxes/ folders, e.g.
    results/mask_generation/fip/plot_461/yolo_sam_v1/pre_refactor/bboxes
"""

import os
import argparse
import torch


def load_boxes(pt_path):
    """Load one image's saved boxes; return an [N,4] tensor (empty [0,4] if none)."""
    t = torch.load(pt_path, weights_only=True)
    if t.numel() == 0:
        return torch.zeros((0, 4))
    return t


def compare_dirs(dir_a, dir_b, atol):
    """Compare every matching *.pt in two bboxes folders; print per-file diffs + a summary."""
    files_a = sorted(f for f in os.listdir(dir_a) if f.endswith(".pt"))
    files_b = sorted(f for f in os.listdir(dir_b) if f.endswith(".pt"))

    print(f"A: {dir_a}  ({len(files_a)} files)")
    print(f"B: {dir_b}  ({len(files_b)} files)")
    if files_a != files_b:
        only_a = set(files_a) - set(files_b)
        only_b = set(files_b) - set(files_a)
        print(f"WARNING: file lists differ — only in A: {len(only_a)}, only in B: {len(only_b)}")

    common = sorted(set(files_a) & set(files_b))
    tot_a = tot_b = n_count_diff = n_coord_diff = 0

    for f in common:
        ta = load_boxes(os.path.join(dir_a, f))
        tb = load_boxes(os.path.join(dir_b, f))
        na, nb = ta.shape[0], tb.shape[0]
        tot_a += na
        tot_b += nb
        if na != nb:
            print(f"  COUNT DIFF {f}: {na} vs {nb}")
            n_count_diff += 1
        elif na > 0 and not torch.allclose(ta, tb, atol=atol):
            print(f"  COORD DIFF {f}: max |Δ| = {(ta - tb).abs().max():.3f} px")
            n_coord_diff += 1

    print("-" * 55)
    print(f"compared files : {len(common)}")
    print(f"total boxes    : A={tot_a}  B={tot_b}  (Δ={tot_a - tot_b:+d})")
    print(f"count diffs    : {n_count_diff}")
    print(f"coord diffs    : {n_coord_diff}  (atol={atol} px)")
    if n_count_diff == 0 and n_coord_diff == 0:
        print("=> IDENTICAL output.")
    elif n_count_diff <= 3 and n_coord_diff == 0:
        print("=> equivalent within GPU non-determinism (a few borderline boxes).")
    else:
        print("=> DIFFERENT — inspect the diffs above.")


def main():
    """Parse the two folder args and run the comparison."""
    ap = argparse.ArgumentParser(description="Compare two mask-generation bboxes/ folders.")
    ap.add_argument("dir_a", help="first bboxes/ folder")
    ap.add_argument("dir_b", help="second bboxes/ folder")
    ap.add_argument("--atol", type=float, default=0.01,
                    help="pixel tolerance for matching box coordinates (default 0.01)")
    args = ap.parse_args()
    compare_dirs(args.dir_a, args.dir_b, args.atol)


if __name__ == "__main__":
    main()
