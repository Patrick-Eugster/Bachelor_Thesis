"""Debug visualiser for the v7 disk-search: shows, per v6 candidate, the SEARCH WINDOW,
every dark blob it considered (with its fill ratio), which passed/failed the gates, the
chosen disk, and the decoded code. Makes the "why did it pick that / miss that" visible.

For each candidate it writes a zoomed crop:
  yellow rect = search window (its size is printed)
  green ellipse + fill = blob that PASSED the gates   (thick green = the CHOSEN disk)
  red ellipse + fill   = blob that FAILED a gate
  text = decoded id (or 'no decode')

Usage:
  python src/preprocessing/debug_cct_window.py field=field_A plot=20250609 images=IMG_20250609_112221,IMG_20250609_112220
"""
import os
import sys
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_markers_v6 import build_template_bank, detect_one          # noqa: E402
from cct_forced_decode import find_disk_at, decode_at_center           # noqa: E402
from detect_markers_v7_cct import decode_cfg_for                        # noqa: E402


@hydra.main(version_base=None, config_path="../../configs",
            config_name="preprocessing/detect_markers_v7_cct")
def main(cfg: DictConfig):
    """Render per-candidate debug crops showing the search window + considered blobs + choice."""
    image_dir = os.path.join(cfg.source_path, cfg.image_subdir)
    out_dir = os.path.join(cfg.source_path, "marker_vis_v7_debug")
    os.makedirs(out_dir, exist_ok=True)
    names = [s.strip() for s in str(cfg.get("images", "")).split(",") if s.strip()]
    if not names:
        print("pass images=NAME1,NAME2 (without .jpg)")
        return

    first = cv2.imread(os.path.join(image_dir, names[0] + ".jpg"))
    W0 = first.shape[1]
    work_scale = min(1.0, cfg.match_max_width / float(W0)) if cfg.match_max_width > 0 else 1.0
    templates = build_template_bank(cfg, work_scale)

    for name in names:
        bgr = cv2.imread(os.path.join(image_dir, name + ".jpg"))
        if bgr is None:
            print(f"skip {name}")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cands = detect_one(bgr, templates, work_scale, cfg)
        print(f"\n{name}: v6 proposed {len(cands)} candidate(s)")
        for ci, d in enumerate(cands):
            cx, cy = d["center"]
            dcfg = decode_cfg_for(cfg, d["fid_radius"])
            sr = dcfg["search_r"]
            blobs = []
            find_disk_at(gray, cx, cy, sr, dcfg, debug=blobs)
            code, info = decode_at_center(bgr, cx, cy, dcfg, N=cfg.n_bits, color=cfg.mark_color)
            chosen = info.get("disk_center")
            print(f"  cand {ci}: proposal=({cx:.0f},{cy:.0f}) r={d['fid_radius']:.0f} "
                  f"window=±{sr:.0f}px  blobs={len(blobs)}  -> code={code}")

            # crop a bit larger than the window for context
            pad = int(sr * 1.25)
            x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
            x1, y1 = min(bgr.shape[1], int(cx + pad)), min(bgr.shape[0], int(cy + pad))
            crop = bgr[y0:y1, x0:x1].copy()
            sh = (x0, y0)
            # search window rectangle (yellow)
            cv2.rectangle(crop, (int(cx - sr - sh[0]), int(cy - sr - sh[1])),
                          (int(cx + sr - sh[0]), int(cy + sr - sh[1])), (0, 255, 255), 2)
            cv2.drawMarker(crop, (int(cx - sh[0]), int(cy - sh[1])), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 18, 2)   # v6's proposal
            for b in blobs:
                col = (0, 220, 0) if b["passed"] else (0, 0, 255)
                ecen = (int(b["center"][0] - sh[0]), int(b["center"][1] - sh[1]))
                is_chosen = chosen and abs(b["center"][0] - chosen[0]) < 2 and abs(b["center"][1] - chosen[1]) < 2
                th = 4 if is_chosen else 2
                cv2.ellipse(crop, (ecen, (int(b["axes"][0]), int(b["axes"][1])), b["angle"]),
                            col, th)
                cv2.putText(crop, f"{b['fill']:.2f}", (ecen[0] + 6, ecen[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            label = f"win=+/-{sr:.0f}px  code={code}"
            cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 4)
            cv2.putText(crop, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(out_dir, f"{name}_cand{ci}.png"), crop)
    print(f"\ndebug crops -> {out_dir}/  (yellow=window, green=passed/thick=chosen, red=failed, num=fill)")


if __name__ == "__main__":
    main()
