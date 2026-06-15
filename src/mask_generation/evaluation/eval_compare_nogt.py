"""
eval_compare_nogt.py — GT-free YOLO-vs-SAHI agreement (runs on FIP AND phone).

When there is no manual GT (phone has none, and on FIP this is a GT-free cross-check), you can't
say which method is right — only where they agree vs diverge. The 7-region {GT,YOLO,SAHI} Venn
collapses to 3 regions: agree (Y∩S) / YOLO-only (Y\\S) / SAHI-only (S\\Y). Coherence with the
3-way tool: agree = with-GT regions 1+5, YOLO-only = 2+6, SAHI-only = 3+7 (GT is what splits each
into correct-vs-hallucinated). Useful signal: SAHI-only boxes clustering in dense/small regions =
slicing doing its job.

Outputs (under results/mask_generation/{dataset}/evaluation/compare_nogt/{eval_experiment}/):
  - agreement.json        : per-image + total agree / yolo-only / sahi-only counts + agreement rate
  - overlay_agreement/    : green=agree, blue=YOLO-only, magenta=SAHI-only
  - regions/<name>/        : single-region images (only if overlay_mode = singles/both; agree = bulk, no single)
  - config.yaml

Run:  python src/mask_generation/evaluation/eval_compare_nogt.py dataset=phone overlay_mode=both
"""

import glob
import os
import json
import shutil
import datetime
import yaml
import numpy as np

import hydra
from omegaconf import DictConfig

from eval_yolo_boxes import load_pred_boxes, get_eval_experiment
import compare_common as cc


# region keys / labels / colors (only 3 without GT)
REGIONS = ['agree', 'yolo_only', 'sahi_only']
LABELS = {'agree': 'agree (YOLO∩SAHI)', 'yolo_only': 'YOLO-only', 'sahi_only': 'SAHI-only'}
COLORS = {'agree': (40, 200, 40), 'yolo_only': (40, 120, 255), 'sahi_only': (220, 40, 200)}
SINGLE_REGIONS = ['yolo_only', 'sahi_only']   # agree is the bulk → never a single image


def bbox_dir(cfg, method, experiment, plot_name):
    """Folder holding a method's per-image 4-col boxes for one plot."""
    return os.path.join(cfg.dataset.result_dir_masks, plot_name, method, experiment, 'bboxes')


def find_image(input_plot_dir, stem):
    """Locate the undistorted image for a stem (png first, then jpg). Returns None if absent."""
    for ext in ('.png', '.jpg'):
        p = os.path.join(input_plot_dir, 'images', stem + ext)
        if os.path.exists(p):
            return p
    return None


def evaluate(cfg):
    """Compare YOLO vs SAHI boxes on every image that both methods produced, draw the agreement
    overlay(s), and write agreement.json + config.yaml."""
    iou_thr = cfg.matching_iou_threshold
    eval_dir = os.path.join(cfg.dataset.result_dir_masks, "evaluation", "compare_nogt", get_eval_experiment(cfg))
    overlay_dir = os.path.join(eval_dir, 'overlay_agreement')
    regions_dir = os.path.join(eval_dir, 'regions')
    for d in [eval_dir, overlay_dir, regions_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    print(f"\n{'=' * 64}\n YOLO vs SAHI — agreement (no GT)\n{'=' * 64}")
    print(f" YOLO boxes:  {cfg.yolo_method}/{cfg.yolo_experiment}")
    print(f" SAHI boxes:  {cfg.sahi_method}/{cfg.sahi_experiment}")
    print(f" dataset: {cfg.dataset.name}   match IoU: {iou_thr}   overlay_mode: {cfg.overlay_mode}")
    print(f"{'=' * 64}")

    plot_dirs = sorted(glob.glob(os.path.join(cfg.dataset.input_dir, cfg.plot_glob)))
    per_image = []
    tot = {r: 0 for r in REGIONS}
    for input_plot_dir in plot_dirs:
        if not os.path.isdir(os.path.join(input_plot_dir, 'images')):
            continue
        plot_name = os.path.relpath(input_plot_dir, cfg.dataset.input_dir)
        plot_safe = plot_name.replace(os.sep, '_')

        y_dir = bbox_dir(cfg, cfg.yolo_method, cfg.yolo_experiment, plot_name)
        s_dir = bbox_dir(cfg, cfg.sahi_method, cfg.sahi_experiment, plot_name)
        if not (os.path.isdir(y_dir) and os.path.isdir(s_dir)):
            print(f"[SKIP] {plot_name}: missing YOLO or SAHI bboxes folder.")
            continue
        # only images both methods produced boxes for
        y_stems = {f[:-3] for f in os.listdir(y_dir) if f.endswith('.pt')}
        s_stems = {f[:-3] for f in os.listdir(s_dir) if f.endswith('.pt')}
        common = sorted(y_stems & s_stems)
        if cfg.limit_images > 0:
            common = common[:cfg.limit_images]

        for stem in common:
            image_path = find_image(input_plot_dir, stem)
            if image_path is None:
                print(f"[SKIP] {plot_name}/{stem}: no image file.")
                continue
            yolo = load_pred_boxes(os.path.join(y_dir, stem + '.pt'))
            sahi = load_pred_boxes(os.path.join(s_dir, stem + '.pt'))
            cat = cc.categorize_two_sets(yolo, sahi, iou_thr)

            # boxes per region (draw YOLO's box for the agreed pairs)
            region_boxes = {
                'agree':     yolo[[a for a, _, _ in cat['mutual']]] if cat['mutual'] else np.zeros((0, 4), np.float32),
                'yolo_only': yolo[cat['a_only']] if cat['a_only'] else np.zeros((0, 4), np.float32),
                'sahi_only': sahi[cat['b_only']] if cat['b_only'] else np.zeros((0, 4), np.float32),
            }
            counts = {r: len(region_boxes[r]) for r in REGIONS}
            for r in REGIONS:
                tot[r] += counts[r]

            name_tag = f"{plot_safe}_{stem}"
            if cfg.overlay_mode in ('themed', 'both'):
                cc.draw_overlay(image_path,
                                {LABELS[r]: (COLORS[r], region_boxes[r]) for r in REGIONS},
                                os.path.join(overlay_dir, name_tag + '.jpg'))
            if cfg.overlay_mode in ('singles', 'both'):
                for r in SINGLE_REGIONS:
                    rd = os.path.join(regions_dir, r)
                    os.makedirs(rd, exist_ok=True)
                    cc.draw_overlay(image_path, {LABELS[r]: (COLORS[r], region_boxes[r])},
                                    os.path.join(rd, name_tag + '.jpg'))

            per_image.append({'plot': plot_safe, 'stem': stem,
                              'yolo_count': len(yolo), 'sahi_count': len(sahi), **counts})
            print(f"  {plot_name}/{stem}: YOLO {len(yolo)}  SAHI {len(sahi)}  | "
                  f"agree {counts['agree']}  yolo_only {counts['yolo_only']}  sahi_only {counts['sahi_only']}")

    if not per_image:
        print("No images had both YOLO and SAHI boxes — nothing to compare.")
        return

    union = tot['agree'] + tot['yolo_only'] + tot['sahi_only']
    agree_rate = tot['agree'] / union if union else 0.0
    print(f"\n TOTALS ({len(per_image)} images):  agree {tot['agree']}  "
          f"yolo_only {tot['yolo_only']}  sahi_only {tot['sahi_only']}  | agreement rate {agree_rate:.3f}")
    print(f"   (YOLO total {tot['agree'] + tot['yolo_only']}, SAHI total {tot['agree'] + tot['sahi_only']})")

    out = {
        'matching_iou_threshold': iou_thr,
        'methods': {'yolo': f"{cfg.yolo_method}/{cfg.yolo_experiment}",
                    'sahi': f"{cfg.sahi_method}/{cfg.sahi_experiment}"},
        'dataset': cfg.dataset.name,
        'n_images': len(per_image),
        'totals': {**tot, 'yolo_total': tot['agree'] + tot['yolo_only'],
                   'sahi_total': tot['agree'] + tot['sahi_only'], 'agreement_rate': agree_rate},
        'per_image': per_image,
    }
    with open(os.path.join(eval_dir, 'agreement.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(eval_dir, 'config.yaml'), 'w') as f:
        yaml.dump({'experiment': get_eval_experiment(cfg),
                   'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                   'dataset': cfg.dataset.name,
                   'yolo': f"{cfg.yolo_method}/{cfg.yolo_experiment}",
                   'sahi': f"{cfg.sahi_method}/{cfg.sahi_experiment}",
                   'matching_iou_threshold': iou_thr, 'overlay_mode': cfg.overlay_mode},
                  f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved → {os.path.join(eval_dir, 'agreement.json')}\n")
    return out


@hydra.main(version_base=None, config_path="../../../configs/mask_generation", config_name="eval_compare_nogt")
def main(cfg: DictConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
