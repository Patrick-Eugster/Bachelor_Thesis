"""
sahi_merge_debug.py — make SAHI's merge step visible so you can tune it by eye + numbers.

The crux of SAHI is the merge: overlap means the same head is detected in several tiles, and
the merge must collapse those duplicates WITHOUT fusing two genuinely distinct neighbour heads.
The production pipeline only keeps the final merged boxes, so this standalone tool RE-RUNS the
SAHI tile inference (YOLO-on-tiles only, no SAM → seconds per image) and snapshots both stages.
It imports the exact functions the pipeline uses (compute_tile_boxes / load_and_slice /
infer_tiles / merge_preds) — the pre-merge boxes are simply infer_tiles' output before merge_preds.
It does NOT modify the SAHI pipeline.

Per inspected image it writes (under
results/mask_generation/{dataset}/evaluation/sahi_merge_debug/{eval_experiment}/{plot}/):
  - tiles/        : the tile grid + the raw per-tile boxes (spot heads cut at tile seams)
  - before_merge/ : every raw box mapped to the full image (duplicates deliberately visible)
  - after_merge/  : the final boxes SAHI kept
  - clusters/     : each final box's contributing raw boxes share its color
                    (many same-color boxes round one head = good merge; one box spanning two
                    distinct heads = bad over-merge)
And merge_counts.json: N_raw vs N_final (collapsed = N_raw − N_final) + the SAHI knob values,
so you can track collapsing vs overlap_ratio / match_threshold across runs.

Run:  python src/mask_generation/sahi_yolo_sam/sahi_merge_debug.py plot_glob=plot_461 limit_images=1
"""

import os
import sys
import glob
import json
import shutil
import datetime
import colorsys
import yaml
import numpy as np
import torch

import hydra
from omegaconf import DictConfig

from PIL import Image, ImageDraw

# the SAHI pipeline functions (sibling import — this file lives next to sahi_yolo_pipelined.py)
import sahi_yolo_pipelined as sp
from sahi_yolo_pipelined import compute_tile_boxes, load_and_slice, infer_tiles, merge_preds

# draw_overlay (shared drawer) + compute_iou_matrix + eval-experiment naming live in evaluation/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
import compare_common as cc
from eval_yolo_boxes import compute_iou_matrix, get_eval_experiment


def load_yolo(cfg):
    """Load the same YOLO model the SAHI pipeline uses, set conf=nms_floor so tiles keep the full
    confidence range before the merge (exactly as run_yolo_phase_sahi does)."""
    base = os.path.dirname(sp.__file__)
    weights_dir = os.path.join(base, "..", "weights")
    yolo_dir    = os.path.join(base, "..", "yolov5")
    wheat_model = os.path.join(weights_dir, cfg.method.wheat_yolo_model)
    if not os.path.exists(wheat_model):
        raise FileNotFoundError(f"Wheat model not found at {wheat_model}")
    model = torch.hub.load(yolo_dir, 'custom', path=wheat_model, source='local')
    model.conf = cfg.method.conf_threshold_nms_floor
    model.iou  = cfg.method.iou_threshold_nms
    model.classes = list(cfg.method.classes_to_detect)
    return model


def _cluster_color(i):
    """Distinct-ish RGB per index using a golden-ratio hue walk (so adjacent final boxes that
    happen to be near each other rarely get the same color)."""
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.8, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_clusters(image_path, preds, merged, out_path, iou_threshold):
    """Color each final (merged) box and every raw pre-merge box by the final box it collapsed into,
    so you can judge merge quality by eye: a tight group of same-color raw boxes around one head =
    good (duplicates merged); one final box whose cluster straddles two separate heads = bad over-merge.
    Raw boxes that match no final box (max IoU below thr) are drawn thin gray."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    preds4  = np.asarray(preds,  dtype=np.float32).reshape(-1, 5)[:, :4] if len(preds)  else np.zeros((0, 4), np.float32)
    merged4 = np.asarray(merged, dtype=np.float32).reshape(-1, 5)[:, :4] if len(merged) else np.zeros((0, 4), np.float32)

    if len(merged4) == 0 or len(preds4) == 0:
        img.save(out_path, quality=92)
        print(f"  Clusters saved (empty): {out_path}")
        return

    # each raw box → the final box it overlaps most (that's the cluster it belongs to)
    iou = compute_iou_matrix(preds4, merged4)        # (N_raw, N_final)
    assign = iou.argmax(axis=1)
    best   = iou.max(axis=1)

    # final boxes: thick outline in the cluster color
    for j, (x1, y1, x2, y2) in enumerate(merged4):
        draw.rectangle([x1, y1, x2, y2], outline=_cluster_color(j), width=4)
    # raw boxes: thin outline in their final box's color (gray if they matched nothing)
    for i, (x1, y1, x2, y2) in enumerate(preds4):
        if best[i] >= iou_threshold:
            color = _cluster_color(int(assign[i]))
        else:
            color = (150, 150, 150)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

    img.save(out_path, quality=92)
    print(f"  Clusters saved: {out_path}")


def select_images(cfg, plot_dir):
    """Pick which images of a plot to inspect: GT-labeled ones first (if labeled_only and a
    manual_label/ exists), else the sorted image list — capped at limit_images."""
    images_dir = os.path.join(plot_dir, 'images')
    files = sorted(glob.glob(os.path.join(images_dir, '*.png')) + glob.glob(os.path.join(images_dir, '*.jpg')))
    if cfg.labeled_only:
        label_dir = os.path.join(plot_dir, 'manual_label')
        if os.path.isdir(label_dir):
            stems = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}
            labeled = [f for f in files if os.path.splitext(os.path.basename(f))[0] in stems]
            if labeled:
                files = labeled
    return files[:cfg.limit_images] if cfg.limit_images > 0 else files


def run(cfg):
    """Re-run SAHI tile inference + merge on the selected images and write the four debug overlays
    + merge_counts.json."""
    iou_thr = cfg.matching_iou_threshold if 'matching_iou_threshold' in cfg else 0.35
    exp = get_eval_experiment(cfg)
    out_root = os.path.join(cfg.dataset.result_dir_masks, "evaluation", "sahi_merge_debug", exp)
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root)

    knobs = {
        'slice_size': int(cfg.method.sahi_slice_size),
        'overlap_ratio': float(cfg.method.sahi_overlap_ratio),
        'merge': str(cfg.method.sahi_merge),
        'match_metric': str(cfg.method.sahi_match_metric),
        'match_threshold': float(cfg.method.sahi_match_threshold),
        'full_image_pass': bool(cfg.method.sahi_full_image_pass),
    }
    print(f"\n{'=' * 64}\n SAHI MERGE DEBUG\n{'=' * 64}")
    print(f"  knobs: {knobs}")

    model = load_yolo(cfg)

    plot_dirs = sorted(glob.glob(os.path.join(cfg.dataset.input_dir, cfg.plot_glob)))
    rows = []
    for plot_dir in plot_dirs:
        if not os.path.isdir(os.path.join(plot_dir, 'images')):
            continue
        plot_name = os.path.relpath(plot_dir, cfg.dataset.input_dir)
        plot_safe = plot_name.replace(os.sep, '_')
        image_files = select_images(cfg, plot_dir)
        if not image_files:
            continue

        dirs = {sub: os.path.join(out_root, plot_safe, sub)
                for sub in ['tiles', 'before_merge', 'after_merge', 'clusters']}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        for img_path in image_files:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            # re-run the SAHI sequence; preds = PRE-merge, merged = POST-merge
            _, img_np, h, w, crops, offsets = load_and_slice(img_path, cfg)
            tiles  = compute_tile_boxes(w, h, cfg.method.sahi_slice_size, cfg.method.sahi_overlap_ratio)
            preds  = infer_tiles(model, img_np, crops, offsets, w, h, cfg)
            merged = merge_preds(preds, h, w, cfg)
            n_raw, n_final = len(preds), len(merged)

            tile_boxes = np.array([[x0, y0, x1, y1] for (x0, y0, x1, y1) in tiles], dtype=np.float32)
            # 1. tile grid + raw per-tile boxes
            cc.draw_overlay(img_path,
                            {'tile (slice)': ((255, 255, 0), tile_boxes),
                             'raw box':      ((0, 180, 255), preds[:, :4])},
                            os.path.join(dirs['tiles'], stem + '.jpg'), line_width=2)
            # 2. before merge (duplicates visible)
            cc.draw_overlay(img_path, {'raw (pre-merge)': ((255, 80, 80), preds[:, :4])},
                            os.path.join(dirs['before_merge'], stem + '.jpg'), line_width=2)
            # 3. after merge (final boxes)
            cc.draw_overlay(img_path, {'merged': ((40, 200, 40), merged[:, :4] if n_final else merged)},
                            os.path.join(dirs['after_merge'], stem + '.jpg'), line_width=2)
            # 4. clusters (which raw boxes collapsed into which final box)
            draw_clusters(img_path, preds, merged, os.path.join(dirs['clusters'], stem + '.jpg'), iou_thr)

            collapsed = n_raw - n_final
            rows.append({'plot': plot_safe, 'stem': stem, 'n_tiles': len(tiles),
                         'n_raw': n_raw, 'n_final': n_final, 'collapsed': collapsed,
                         'collapsed_ratio': (collapsed / n_raw) if n_raw else 0.0})
            print(f"  {plot_name}/{stem}: tiles {len(tiles)}  N_raw {n_raw}  N_final {n_final}  "
                  f"collapsed {collapsed} ({rows[-1]['collapsed_ratio']*100:.1f}%)")

    with open(os.path.join(out_root, 'merge_counts.json'), 'w') as f:
        json.dump({'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   'knobs': knobs, 'per_image': rows}, f, indent=2)

    # config snapshot (parity with the other evals). No "source experiment" here — this tool
    # re-runs SAHI from the YOLO model + method config, so the reproducibility info is the knobs.
    with open(os.path.join(out_root, 'config.yaml'), 'w') as f:
        yaml.dump({'experiment': exp,
                   'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   'dataset': cfg.dataset.name,
                   'method': cfg.method.name,
                   'plot_glob': cfg.plot_glob,
                   'limit_images': cfg.limit_images,
                   'labeled_only': cfg.labeled_only,
                   'matching_iou_threshold': iou_thr,
                   'sahi_knobs': knobs},
                  f, default_flow_style=False, sort_keys=False)

    # small recap table
    print(f"\n  {'image':<45} {'tiles':>6} {'N_raw':>6} {'N_final':>8} {'collapsed':>10}")
    for r in rows:
        print(f"  {r['plot'] + '/' + r['stem']:<45} {r['n_tiles']:>6} {r['n_raw']:>6} "
              f"{r['n_final']:>8} {r['collapsed']:>6} ({r['collapsed_ratio']*100:.0f}%)")
    print(f"\nSaved → {os.path.join(out_root, 'merge_counts.json')}\n")


@hydra.main(version_base=None, config_path="../../../configs/mask_generation", config_name="sahi_merge_debug")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
