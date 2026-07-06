#!/usr/bin/env python3
"""Probe whether the frustum cull ever drops a camera where a head actually MATCHES.

For the first N heads (in the real seeded processing order) this lifts the head, then compares:
  - cull-kept cameras   = cull_cameras(all, gaussians, obj_used_mask)
  - no-cull matched     = find_match(all cameras, NO cull) -> cameras that match with IoU>thr
A camera that is in "no-cull matched" but NOT in "cull-kept" is a MATCH THE CULL WOULD HAVE DROPPED
-> the cull is non-conservative there (the root-cause bug). If that count is always 0, the cull never
drops a matching view and the phone loss must come from the order-dependent cascade, not the cull.

This reuses run_3d_seg's OWN functions so it exercises the exact code paths. Renders locally on GPU;
writes nothing. Only the INITIAL find_match round is probed (before fine-tune growth) — that's where
conservativeness is cleanest to test.

Usage (same -s/-m/--seg_dir as the real run):
  python src/analysis/probe_cull_conservativeness.py \
     -s input_plots/phone/field_A/20250715 \
     -m results/reconstruction/phone/field_A/20250715/vanilla_3dgs/phone_sahi \
     --seg_dir results/mask_generation/phone/field_A/20250715/sahi_yolo_sam/initial \
     --resolution 1 --eval --iou_threshold 0.5 --probe_n 30
"""
import os
import sys
import random
from argparse import ArgumentParser

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "segmentation_3d"))
from run_3d_seg import (  # noqa: E402  reuse the exact seg code paths
    GaussianModel, Scene, ModelParams, OptimizationParams, PipelineParams,
    opt_label_w_seg, counts_to_obj_labels, cull_cameras, flashsplat_render,
)


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=str, default="-1")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--seg_seed", type=int, default=0)
    parser.add_argument("--probe_n", type=int, default=30)
    args = parser.parse_args()

    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    # --- setup, mirrored from run_3d_seg.training() ---
    gaussians = GaussianModel(dataset.sh_degree)
    try:
        li = int(args.load_iteration)
    except ValueError:
        li = args.load_iteration
    scene = Scene(dataset, gaussians, load_iteration=li, shuffle=False)
    gaussians.training_setup(opt)
    z_mean = torch.mean(gaussians.get_xyz.cpu()[:, 2])
    pts_filter = (gaussians.get_xyz.cpu()[:, 2] < z_mean)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    bbox_cache = {cam.image_name: torch.load(cam.bbox_path) / cam.resolution_scale
                  for cam in viewpoint_stack + scene.getTestCameras().copy()}
    print(f"Loaded {len(gaussians.get_xyz)} gaussians, {len(viewpoint_stack)} train cameras")

    # reconstruct the exact processing order (same as run_3d_seg)
    all_mask_paths = []
    for cam in viewpoint_stack:
        all_mask_paths += cam.mask_paths
    random.seed(args.seg_seed)
    random.shuffle(all_mask_paths)

    EMPTY = {}  # find_match needs a dict (falls back to on-the-fly PNG decode); None would crash
    probed = 0
    total_missed = 0
    heads_with_miss = 0

    for this_mask_path in all_mask_paths:
        if probed >= args.probe_n:
            break
        this_mask_name = os.path.splitext(os.path.basename(this_mask_path))[0]
        this_image_name = this_mask_name[:-4]
        this_cam = next((c for c in viewpoint_stack if c.image_name == this_image_name), None)
        if this_cam is None:
            continue

        # lift this one mask -> the head's gaussians
        all_counts = opt_label_w_seg(gaussians, [this_cam], [this_mask_path], pipe, background,
                                     pts_filter, mask_cache=EMPTY)
        all_obj_labels = counts_to_obj_labels(all_counts)
        if torch.sum(all_obj_labels, dim=1)[1] == 0:
            continue
        obj_used_mask = all_obj_labels[1].bool()

        num_gs = int(obj_used_mask.sum())
        targets = [v for v in viewpoint_stack if v.image_name != this_image_name]
        kept_names = set(c.image_name for c in cull_cameras(targets, gaussians, obj_used_mask))

        # FAST mode: only characterize how much the cull culls (no 83-camera render loop)
        if os.environ.get("WHEAT_PROBE_FAST") == "1":
            probed += 1
            print(f"[{probed:3d}] {this_mask_name}: num_GS={num_gs:5d}  cull kept {len(kept_names):2d}/{len(targets)}",
                  flush=True)
            continue

        # RENDER-ONLY conservativeness test: a camera where the head renders a non-empty blob
        # (alpha>0.5 somewhere — exactly what find_match thresholds as pred_seg) is a camera the cull
        # MUST keep. If such a camera is culled, the cull is non-conservative (could lose a match there).
        rendered_names = []
        for cam in targets:
            with torch.no_grad():
                alpha = flashsplat_render(cam, gaussians, pipe, background,
                                          used_mask=obj_used_mask, inference=True)["alpha"]
                if (alpha.squeeze() > 0.5).any():
                    rendered_names.append(cam.image_name)
        missed = [n for n in rendered_names if n not in kept_names]  # renders a blob but the cull drops it

        probed += 1
        total_missed += len(missed)
        heads_with_miss += 1 if missed else 0
        flag = "   <<< CULL DROPS A NON-EMPTY RENDER" if missed else ""
        print(f"[{probed:3d}] {this_mask_name}: renders a blob in {len(rendered_names):2d} cams | "
              f"cull kept {len(kept_names):2d}/{len(targets)} | dropped-non-empty {len(missed)}{flag}", flush=True)
        for n in missed:
            print(f"        DROPPED non-empty view: {n}", flush=True)

    print(f"\n=== probed {probed} heads ===", flush=True)
    print(f"heads where the cull drops >=1 non-empty-render view: {heads_with_miss}")
    print(f"total dropped-non-empty views (conservativeness violations): {total_missed}")
    if total_missed == 0:
        print("=> CULL IS CONSERVATIVE on these heads (never dropped a camera with a non-empty render).")
        print("   => the phone loss is the order/cascade effect, NOT the cull dropping visible views.")
    else:
        print("=> CULL DROPPED NON-EMPTY RENDERS => it is NON-conservative => this is the root-cause bug.")


if __name__ == "__main__":
    main()
