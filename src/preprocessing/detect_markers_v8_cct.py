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
import multiprocessing as mp
import os
import shutil
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


# --- parallel pass-1 helpers -------------------------------------------------
# Each image decodes fully independently, so we split the work across N worker PROCESSES (not threads —
# the decode is CPU-bound Python, the GIL would serialise threads). The shared read-only inputs
# (template bank, work scale, cfg, image dir) are handed to each worker ONCE via the pool initializer
# so the big template bank isn't re-pickled per image.
_WORKER = {}


def _init_worker(image_dir, templates, work_scale, cfg):
    """Runs once when a worker process starts: stash the shared read-only inputs as module globals.
    Also pins OpenCV to 1 thread per process so N processes don't each spawn their own OpenCV threads
    and oversubscribe the cores."""
    cv2.setNumThreads(1)
    _WORKER["image_dir"] = image_dir
    _WORKER["templates"] = templates
    _WORKER["work_scale"] = work_scale
    _WORKER["cfg"] = cfg


def _decode_one(f):
    """Worker task: decode ONE image, return (filename, dets) — or (filename, None) if unreadable.
    Module-level so multiprocessing can pickle and ship it to a worker."""
    bgr = cv2.imread(os.path.join(_WORKER["image_dir"], f))
    if bgr is None:
        return f, None
    return f, detect_and_decode(bgr, _WORKER["templates"], _WORKER["work_scale"], _WORKER["cfg"])


def _iter_decode_results(files, image_dir, templates, work_scale, cfg, num_workers):
    """Yield (filename, dets) for every image. num_workers<=1 → serial (current behaviour, easiest to
    debug / fully deterministic); else a process Pool of num_workers decodes images in parallel."""
    if num_workers <= 1:
        _init_worker(image_dir, templates, work_scale, cfg)
        for f in files:
            yield _decode_one(f)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(num_workers, initializer=_init_worker,
                      initargs=(image_dir, templates, work_scale, cfg)) as pool:
            # imap keeps file order so the progress print stays sensible; chunksize amortises IPC
            yield from pool.imap(_decode_one, files, chunksize=4)


# --- parallel pass-2 helpers (overlay drawing + PNG saving) ------------------
# Same idea as pass 1: drawing each overlay + writing its PNG is fully independent per image, so split
# it across the same N worker processes. The per-image detections travel with the task (small dicts),
# the rest (dirs, overlay width) is handed once via the initializer.
_DRAW = {}


def _init_draw_worker(image_dir, vis_dir, overlay_max_width):
    """Runs once per draw-worker process: stash the shared paths + overlay width; 1 OpenCV thread each."""
    cv2.setNumThreads(1)
    _DRAW["image_dir"] = image_dir
    _DRAW["vis_dir"] = vis_dir
    _DRAW["overlay_max_width"] = overlay_max_width


def _draw_one(item):
    """Worker task: read one image, draw its overlay, save the PNG. item = (filename, dets).
    Returns (filename, n_dets) or (filename, None) if unreadable."""
    f, dets = item
    bgr = cv2.imread(os.path.join(_DRAW["image_dir"], f))
    if bgr is None:
        return f, None
    cv2.imwrite(os.path.join(_DRAW["vis_dir"], f), draw_overlay(bgr, dets, _DRAW["overlay_max_width"]))
    return f, len(dets)


def _iter_draw_results(items, image_dir, vis_dir, overlay_max_width, num_workers):
    """Yield (filename, n_dets) for every (filename, dets) item. num_workers<=1 → serial; else a Pool.
    Order doesn't matter here (we only tally counts), so imap_unordered is fine."""
    if num_workers <= 1:
        _init_draw_worker(image_dir, vis_dir, overlay_max_width)
        for it in items:
            yield _draw_one(it)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(num_workers, initializer=_init_draw_worker,
                      initargs=(image_dir, vis_dir, overlay_max_width)) as pool:
            yield from pool.imap_unordered(_draw_one, items, chunksize=4)


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
    # wipe the overlay folder first so a re-run on a SMALLER image set can't leave stale PNGs from a
    # previous (larger) run lying around — we want the folder to mirror exactly this run's images.
    shutil.rmtree(vis_dir, ignore_errors=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"v8 detect+decode (concentric centre) over {len(files)} images from {image_dir}")
    print(f"Overlays → {vis_dir}/")

    # pass 1: decode every image (in parallel across worker processes), tally ID view counts.
    # num_workers default 8 (≈ physical cores on the Ryzen 7700X3D; the 16 SMT threads add little on
    # this FP-heavy NCC/decode work and just thrash cache). Capped to the image count.
    num_workers = max(1, min(int(cfg.get("num_workers", 8)), len(files)))
    print(f"v8 pass 1: decoding {len(files)} images with {num_workers} worker(s)")
    per_image = {}
    id_views = {}
    for i, (f, dets) in enumerate(
            _iter_decode_results(files, image_dir, templates, work_scale, cfg, num_workers)):
        if dets is None:            # unreadable image
            continue
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

    # pass 2: draw overlays from the filtered detections only (parallel, same worker count as pass 1)
    items = [(f, per_image[f]) for f in files if f in per_image]
    counts = []
    for f, n in _iter_draw_results(items, image_dir, vis_dir, cfg.overlay_max_width, num_workers):
        if n is not None:                 # skip unreadable images
            counts.append(n)

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
