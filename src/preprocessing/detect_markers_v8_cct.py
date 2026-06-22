"""Marker DETECTOR v8 — v7 pipeline + CONCENTRIC-CONSENSUS centre finding.

v7 decodes only when it can find the solid central DISK at v6's candidate. It therefore
throws the whole marker away when v6 lands on a code ARC, or when the disk is occluded by
wheat — even though the marker is clearly there.

v8 fixes that. The disk and all arcs are CONCENTRIC (share one centre = the white dot), so
`find_center_concentric` recovers the centre from whatever parts survive:
  * solid disk present  -> use it (sharpest centre, == v7 behaviour);
  * disk occluded/absent -> fit the code RING to the >=2 arcs and derive the disk-equivalent
    ellipse (ring / 2.5) at the ring centre.
Everything downstream (rectify -> decode -> ID) is unchanged; only the centre finder differs.

READ-ONLY: overlays -> marker_vis_v8/, detections+IDs -> logs/marker_detections_v8.json.

Usage:
    python src/preprocessing/detect_markers_v8_cct.py field=field_A plot=20250609
    python src/preprocessing/detect_markers_v8_cct.py field=field_A plot=20250609 limit=5
"""

import json
import os
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from detect_markers_v6 import build_template_bank, detect_one
from cct_forced_decode import decode_at_center, find_center_concentric
from detect_markers_v7_cct import DEGENERATE, decode_cfg_for, id_color, draw_overlay
import marker_codes


def detect_and_decode(bgr, templates, work_scale, cfg):
    """v6 proposes centres -> concentric finder locates the disk (or reconstructs it from the
    arcs) -> decode -> keep valid codes -> one best detection per id per image."""
    cands = detect_one(bgr, templates, work_scale, cfg)
    decoded = []
    for d in cands:
        cx, cy = d["center"]
        code, info = decode_at_center(bgr, cx, cy, decode_cfg_for(cfg, d["fid_radius"]),
                                      N=cfg.n_bits, color=cfg.mark_color,
                                      finder=find_center_concentric)
        if code is None or code in DEGENERATE:
            continue
        center = info.get("disk_center", d["center"])
        decoded.append({"center": [round(center[0], 2), round(center[1], 2)], "id": int(code),
                        "score": d["score"], "fid_radius": d["fid_radius"]})
    best = {}
    for d in decoded:
        if d["id"] not in best or d["score"] > best[d["id"]]["score"]:
            best[d["id"]] = d
    return list(best.values())


@hydra.main(version_base=None, config_path="../../configs",
            config_name="preprocessing/detect_markers_v8_cct")
def main(cfg: DictConfig):
    """Run v8 (v6 proposal + concentric-consensus centre + CCTDecode) over a plot."""
    print("--- detect_markers_v8_cct config ---")
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
    print(f"v8 detect+decode (concentric centre) over {len(files)} images from {image_dir}")
    print(f"Overlays → {vis_dir}/")

    # pass 1: decode every image, tally how many views each ID is seen in
    per_image = {}
    id_views = {}
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(image_dir, f))
        if bgr is None:
            continue
        dets = detect_and_decode(bgr, templates, work_scale, cfg)
        per_image[f] = dets
        for d in dets:
            id_views[d["id"]] = id_views.get(d["id"], 0) + 1
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  [{i+1}/{len(files)}] {f:<32} → {len(dets)} raw marker(s) "
                  f"{[d['id'] for d in dets]}")

    # ID filter: drop junk IDs before drawing overlays. 'manifest' keeps only the deployed codes
    # (the principled default, sourced from the spec PDF); 'view' = legacy top-k heuristic.
    id_views_raw = dict(id_views)
    dropped = set()
    per_image_dropped = {}
    if cfg.id_filter and cfg.id_filter != "none":
        manifest = list(cfg.plot_manifest) if cfg.plot_manifest else []
        kept = marker_codes.kept_ids(id_views, cfg.id_filter, manifest=manifest,
                                     keep_top_k=cfg.keep_top_k, min_views=cfg.min_views)
        per_image, per_image_dropped = marker_codes.split_detections(per_image, kept)
        dropped = {i for i in id_views if i not in kept}
        id_views = {i: id_views_raw[i] for i in kept}
        extra = f", manifest={sorted(manifest)}" if cfg.id_filter == "manifest" else \
                f", keep_top_k={cfg.keep_top_k}"
        print(f"\nID filter (mode={cfg.id_filter}, min_views={cfg.min_views}{extra}): "
              f"kept {sorted(kept)}")
        if dropped:
            drp = sorted(dropped, key=lambda k: -id_views_raw[k])
            print(f"  dropped {len(dropped)} non-manifest/junk ID(s): "
                  f"{[(i, id_views_raw[i]) for i in drp]}")

    # pass 2: draw overlays from the filtered detections only
    counts = []
    for f in files:
        if f not in per_image:
            continue
        bgr = cv2.imread(os.path.join(image_dir, f))
        if bgr is None:
            continue
        dets = per_image[f]
        counts.append(len(dets))
        cv2.imwrite(os.path.join(vis_dir, f), draw_overlay(bgr, dets, cfg.overlay_max_width))

    counts = np.array(counts) if counts else np.array([0])
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("    MARKER DETECT+DECODE v8 (concentric centre) SUMMARY")
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
        "id_filter": {"mode": cfg.id_filter, "min_views": cfg.min_views,
                      "keep_top_k": cfg.keep_top_k,
                      "manifest": list(cfg.plot_manifest) if cfg.plot_manifest else [],
                      "dropped_ids": sorted(dropped)},
        "id_views_raw": id_views_raw,
        "id_views": id_views,
        "per_image_dropped": per_image_dropped,
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
