"""Stage A+B marker DETECTOR, VERSION 7 (v6 proposal + CCTDecode forced-centre decode).

The full pipeline the marker work has been building toward:

    v6 (real-template NCC + fiducial-snap)  →  candidate marker centres
                       ↓
    decode_at_center() at each candidate    →  validate + read the 12-bit ID
                       ↓
    keep only candidates that yield a valid (non-degenerate) code

The decode does TWO jobs at once:
  1. VALIDATION — a real coded marker yields a valid codeword; wheat / arcs / NCC false
     positives yield nothing → v6's ~extras are dropped for free (the decode is a checksum).
  2. ID — the decoded code is the marker's identity, consistent across views (Phase 1 proved
     forcing the centre gives distinct, stable codes per marker).

Why this beats v1-v6: those could only *localise*; they had no way to reject a false plate or
assign an identity. The code does both. Phase 0/1 showed stock CCTDecode fails because of its
blob search; here v6 supplies the centre and decode_at_center reads only the disk there.

READ-ONLY: overlays → marker_vis_v7/, detections+IDs → logs/marker_detections_v7.json.
Prereq: the real template (make_fiducial_template.py) — same as v6.

Usage:
    python src/preprocessing/detect_markers_v7_cct.py field=field_A plot=20250609
    python src/preprocessing/detect_markers_v7_cct.py field=field_A plot=20250609 limit=5
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# reuse v6's proposal stage wholesale, and the forced-centre decoder
from detect_markers_v6 import build_template_bank, detect_one
from cct_forced_decode import decode_at_center, DEFAULT_CFG

DEGENERATE = {0, 4095, 2047}     # artifact codes (no bits / all bits / disk-ish)

# distinct BGR colours per decoded id (cycled) so the same marker keeps its colour across views
ID_COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255),
             (255, 0, 255), (255, 200, 0), (0, 128, 255), (200, 0, 200)]


def decode_cfg_for(cfg, fid_radius):
    """Build the decode_at_center cfg for one candidate. The disk-search window is SIZE-RELATIVE
    (× the candidate's fiducial radius) — NOT a hardcoded px distance — so it auto-scales with the
    marker's apparent size; the fill-ratio gate (size-independent) is what rejects arcs inside it."""
    c = dict(DEFAULT_CFG)
    c["search_r"] = max(cfg.decode_search_min, fid_radius * cfg.decode_search_factor)
    c["disk_min_fill"] = cfg.decode_min_fill
    c["require_valid_cct"] = cfg.decode_require_valid_cct
    return c


def detect_and_decode(bgr, templates, work_scale, cfg):
    """v6 proposes centres → decode each → keep valid codes → one best detection per id."""
    cands = detect_one(bgr, templates, work_scale, cfg)   # v6 candidate centres
    decoded = []
    for d in cands:
        cx, cy = d["center"]
        code, info = decode_at_center(bgr, cx, cy, decode_cfg_for(cfg, d["fid_radius"]),
                                      N=cfg.n_bits, color=cfg.mark_color)
        if code is None or code in DEGENERATE:
            continue                                      # not a real coded marker → drop
        # report the centre of the disk we actually decoded (true fiducial centre), not v6's
        center = info.get("disk_center", d["center"])
        decoded.append({"center": [round(center[0], 2), round(center[1], 2)], "id": int(code),
                        "score": d["score"], "fid_radius": d["fid_radius"]})
    # one detection per id per image (same marker can't appear twice) — keep the strongest
    best = {}
    for d in decoded:
        if d["id"] not in best or d["score"] > best[d["id"]]["score"]:
            best[d["id"]] = d
    return list(best.values())


def id_color(code):
    return ID_COLORS[code % len(ID_COLORS)]


def draw_overlay(bgr, dets, max_width):
    """Ring (sized to the marker) + centre dot + decoded id (dark outline so it reads on
    any background). Annotations scale with image size so they're visible on 4032 px photos."""
    vis = bgr.copy()
    H, W = vis.shape[:2]
    base = max(H, W) / 1000.0                       # scale factor for thickness/font
    for d in dets:
        cx, cy = int(d["center"][0]), int(d["center"][1])
        col = id_color(d["id"])
        ring = int(max(70, d["fid_radius"] * 2.8))   # encircle the whole marker, not just the disk
        cv2.circle(vis, (cx, cy), ring, col, int(3 * base))
        cv2.drawMarker(vis, (cx, cy), col, cv2.MARKER_CROSS, int(ring * 0.5), int(3 * base))
        txt = f"id={d['id']}"
        org = (cx + ring + 6, cy)
        fs = 1.4 * base
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 40, 40), int(7 * base), cv2.LINE_AA)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, int(3 * base), cv2.LINE_AA)
    if max_width > 0 and W > max_width:
        s = max_width / float(W)
        vis = cv2.resize(vis, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    return vis


@hydra.main(version_base=None, config_path="../../configs",
            config_name="preprocessing/detect_markers_v7_cct")
def main(cfg: DictConfig):
    """Run v7 (v6 proposal + CCTDecode forced-centre decode) over a plot."""
    print("--- detect_markers_v7_cct config ---")
    print(OmegaConf.to_yaml(cfg))
    print("------------------------------------")
    t_start = time.time()

    image_dir = os.path.join(cfg.source_path, cfg.image_subdir)
    if not os.path.isdir(image_dir):
        print(f"ERROR: image dir not found: {image_dir}")
        return
    files = sorted(f for f in os.listdir(image_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if cfg.limit and cfg.limit > 0:
        files = files[:cfg.limit]
    if not files:
        print(f"ERROR: no images found in {image_dir}")
        return

    first = cv2.imread(os.path.join(image_dir, files[0]))
    W0 = first.shape[1]
    work_scale = min(1.0, cfg.match_max_width / float(W0)) if cfg.match_max_width > 0 else 1.0
    templates = build_template_bank(cfg, work_scale)
    print(f"Built {len(templates)} template scales; matching at {work_scale:.3f}×.")

    vis_dir = os.path.join(cfg.source_path, cfg.output_vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"v7 detect+decode over {len(files)} images from {image_dir}")
    print(f"Overlays → {vis_dir}/")

    per_image = {}
    counts = []
    id_views = {}                       # id -> how many images it was seen in
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(image_dir, f))
        if bgr is None:
            continue
        dets = detect_and_decode(bgr, templates, work_scale, cfg)
        per_image[f] = dets
        counts.append(len(dets))
        for d in dets:
            id_views[d["id"]] = id_views.get(d["id"], 0) + 1
        cv2.imwrite(os.path.join(vis_dir, f), draw_overlay(bgr, dets, cfg.overlay_max_width))
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {f:<32} → {len(dets)} marker(s) "
                  f"{[d['id'] for d in dets]}")

    counts = np.array(counts) if counts else np.array([0])
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("        MARKER DETECT+DECODE v7 (CCTDecode) SUMMARY")
    print("=" * 60)
    print(f"{'Plot:':<30} {cfg.field}/{cfg.plot}")
    print(f"{'Images processed:':<30} {len(per_image)}")
    print(f"{'markers/image mean:':<30} {counts.mean():.2f}  (max {counts.max()})")
    print(f"{'images with 0:':<30} {int(np.sum(counts == 0))}")
    print(f"{'total decoded markers:':<30} {int(counts.sum())}")
    print("-" * 60)
    print("decoded IDs (id: #views) — distinct stable IDs = success:")
    for code in sorted(id_views, key=lambda k: -id_views[k]):
        print(f"    id {code:>4}: seen in {id_views[code]} views")
    print("-" * 60)
    m, s = divmod(int(elapsed), 60)
    print(f"{'TOTAL TIME:':<30} {m}m {s}s")
    print("=" * 60 + "\n")

    report = {
        "field": cfg.field, "plot": cfg.plot, "image_subdir": cfg.image_subdir,
        "n_images": len(per_image), "expected_markers": cfg.expected_markers,
        "id_views": id_views,
        "counts": {"mean": float(counts.mean()), "max": int(counts.max()),
                   "n_zero": int(np.sum(counts == 0)), "total": int(counts.sum())},
        "per_image": per_image, "elapsed_s": elapsed,
    }
    out_path = os.path.join(cfg.source_path, cfg.output_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Detections+IDs JSON written to {out_path}\n")


if __name__ == "__main__":
    main()
