"""SAM3 text-prompt DIAGNOSTIC probe — conf sweep on the GT images, full-frame and/or tiled, eyeball only.

Purpose: the first text smoke ran ONE config (full frame @ imgsz 1008, conf 0.10) and looked terrible.
Before calling text-PCS a dead end we sweep the two suspects cheaply, on the SAME images we have GT for:
  (1) CONFIDENCE — sweep conf 0.10..0.50 (step 0.05) and watch how the overlays change.
  (2) RESOLUTION — full-frame downscales a 4032px phone image ~4x into the imgsz encode, so heads vanish.
      "fake tiling" = cut the image into an NxN grid and run each piece at ~native res. NO merge/dedup —
      only to eyeball whether the tiles resolve heads better than the whole frame.

Run it in two phases (matches how you asked to look at it):
  # 1) full-frame conf sweep on every GT image (phone 6 sessions + every FIP plot)
  python src/analysis/sam3_text_probe.py --auto all --mode full  --fp32
  # 2) then the tiled conf sweep
  python src/analysis/sam3_text_probe.py --auto all --mode tiles --fp32 --tiles 2

Or one dataset at a time: --auto phone  /  --auto fip.
GT images are auto-discovered (the ONE labeled image per plot/session):
  phone: input_plots/phone/<field>/<date>/manual_label/<stem>_sets/  -> images/<stem>.*
  fip:   input_plots/fip/<plot>/manual_label/<stem>.txt              -> images/<stem>.*

Overlays + summary.json land under docs/analysis_results/sam3_text_probe/<dataset>/<plot>/<stem>/.
Runs on Euler (40GB A100), FP32 with --fp32 (drop it for FP16 ~13GB on a 24GB card).
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
from PIL import Image


# full candidate vocabulary for the head — cheap to sweep (image encoded once per tile, each phrase is just
# another decode). Includes the ones that fired before ("wheat spike"/"wheat") AND the ones that returned 0
# ("wheat head"/"wheat ear") re-run to be sure, plus botanical synonyms (spike/ear/inflorescence) and
# colloquial ones (grain head/seed head).
DEFAULT_PHRASES = [
    "wheat spike", "spike", "wheat head", "wheat ear", "ear", "ear of wheat", "head of wheat",
    "wheat", "wheat inflorescence", "inflorescence", "grain head", "seed head", "cereal spike",
]
DEFAULT_CONFS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def discover_gt_images(which):
    """Find the ONE GT-labeled image per plot/session for phone and/or fip. Returns a list of
    {dataset, plot, stem, img} — the same GT images the instance eval scores, so the eyeball is on
    representative, GT-covered frames."""
    items = []
    if which in ("phone", "all"):
        for sets_dir in sorted(glob.glob("input_plots/phone/*/*/manual_label/*_sets")):
            stem = os.path.basename(sets_dir)[: -len("_sets")]
            plot_dir = os.path.dirname(os.path.dirname(sets_dir))       # .../<field>/<date>
            plot = os.path.relpath(plot_dir, "input_plots/phone").replace(os.sep, "_")
            hits = glob.glob(os.path.join(plot_dir, "images", stem + ".*"))
            if hits:
                items.append({"dataset": "phone", "plot": plot, "stem": stem, "img": hits[0]})
    if which in ("fip", "all"):
        for txt in sorted(glob.glob("input_plots/fip/*/manual_label/*.txt")):
            stem = os.path.basename(txt)[:-4]                           # strip .txt -> image stem
            plot_dir = os.path.dirname(os.path.dirname(txt))           # .../<plot>
            plot = os.path.relpath(plot_dir, "input_plots/fip")
            hits = glob.glob(os.path.join(plot_dir, "images", stem + ".*"))
            if hits:
                items.append({"dataset": "fip", "plot": plot, "stem": stem, "img": hits[0]})
    return items


def build_predictor(weight, fp32, imgsz):
    """Build a SAM3SemanticPredictor (text/concept interface). conf is swept later via args.conf."""
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = {"model": weight, "task": "segment", "mode": "predict",
                 "conf": 0.10, "save": False, "verbose": False,
                 "project": "runs_sam3_probe"}   # writable project dir (avoids /cluster/home/runs perm error)
    if imgsz:
        overrides["imgsz"] = imgsz
    if not fp32:
        overrides["quantize"] = 16
    return SAM3SemanticPredictor(overrides=overrides)


def result_masks(res):
    """Pull (masks NxHxW uint8, boxes Nx4, conf N) out of one ultralytics Result."""
    r = res[0] if isinstance(res, (list, tuple)) else res
    masks = np.zeros((0, 0, 0), dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)
    conf = np.zeros((0,), dtype=np.float32)
    if getattr(r, "masks", None) is not None and r.masks.data is not None and len(r.masks.data):
        masks = (r.masks.data.detach().cpu().numpy() > 0.5).astype(np.uint8)
    if getattr(r, "boxes", None) is not None and len(r.boxes):
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        conf = r.boxes.conf.detach().cpu().numpy()
    return masks, boxes, conf


def make_overlay(img_bgr, masks, boxes, label):
    """Green mask union + red boxes + a count label. Just for eyeballing."""
    vis = img_bgr.copy()
    H, W = img_bgr.shape[:2]
    if len(masks):
        union = np.zeros((H, W), dtype=bool)
        for m in masks:
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            union |= m.astype(bool)
        green = np.zeros_like(vis); green[union] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, green, 0.45, 0)
    for b in boxes.astype(int):
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cv2.putText(vis, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return vis


def tiles_of(img_rgb, n):
    """Cut img into an n x n grid of NON-overlapping pieces. Yields (tag, y0, x0, tile_rgb).
    tag 'full' when n<=1. Just a plain cut — no overlap, no merge (diagnostic only)."""
    H, W = img_rgb.shape[:2]
    if n <= 1:
        yield "full", 0, 0, img_rgb
        return
    ys = [int(round(i * H / n)) for i in range(n + 1)]
    xs = [int(round(j * W / n)) for j in range(n + 1)]
    for r in range(n):
        for c in range(n):
            y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
            yield f"t{r}{c}", y0, x0, img_rgb[y0:y1, x0:x1].copy()


def modes_for(img_rgb, mode, n):
    """Which pieces to run for this --mode: 'full' -> whole frame only; 'tiles' -> the NxN grid only;
    'both' -> full frame + the grid."""
    out = []
    if mode in ("full", "both"):
        out += list(tiles_of(img_rgb, 1))
    if mode in ("tiles", "both") and n > 1:
        out += list(tiles_of(img_rgb, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", choices=["phone", "fip", "all"], default="all",
                    help="auto-discover the GT image per plot/session for this dataset")
    ap.add_argument("--images", nargs="+", default=None,
                    help="manual image list (overrides --auto); each treated as dataset=manual")
    ap.add_argument("--mode", choices=["full", "tiles", "both"], default="both",
                    help="full = full-frame sweep only; tiles = tiled sweep only; both = full + tiles")
    ap.add_argument("--phrases", nargs="+", default=DEFAULT_PHRASES)
    ap.add_argument("--conf", nargs="+", type=float, default=DEFAULT_CONFS)
    ap.add_argument("--tiles", type=int, default=2, help="n -> n x n grid (2 = 4 pieces)")
    ap.add_argument("--weight", default="src/mask_generation/weights/sam3.pt")
    ap.add_argument("--out", default="docs/analysis_results/sam3_text_probe")
    ap.add_argument("--imgsz", type=int, default=1008, help="SAM3 encode size (multiple of 14)")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    if args.images:
        items = [{"dataset": "manual", "plot": "manual", "stem": os.path.splitext(os.path.basename(p))[0],
                  "img": p} for p in args.images]
    else:
        items = discover_gt_images(args.auto)
    if not items:
        print("No GT images found. Check --auto / --images."); return

    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "_tmp"); os.makedirs(tmp, exist_ok=True)
    print(f"SAM3 text probe | mode={args.mode} | {len(items)} GT image(s) | conf {args.conf} | "
          f"tiles={args.tiles if args.mode != 'full' else '-'} | phrases={args.phrases}")
    predictor = build_predictor(args.weight, args.fp32, args.imgsz)

    summary = []
    for it in items:
        img_path = it["img"]
        if not os.path.exists(img_path):
            print(f"[skip] missing {img_path}"); continue
        stem = it["stem"]
        out_dir = os.path.join(args.out, it["dataset"], it["plot"], stem)
        os.makedirs(out_dir, exist_ok=True)
        img_rgb = np.array(Image.open(img_path).convert("RGB"))       # PIL: silent on Samsung phone JPEGs
        print(f"\n[{it['dataset']}/{it['plot']}/{stem}]  {img_rgb.shape[1]}x{img_rgb.shape[0]}")
        for tag, y0, x0, tile_rgb in modes_for(img_rgb, args.mode, args.tiles):
            tile_bgr = tile_rgb[:, :, ::-1].copy()
            tile_path = os.path.join(tmp, f"{it['dataset']}_{it['plot']}_{stem}_{tag}.png")
            cv2.imwrite(tile_path, tile_bgr)          # set_image reads a path (matches the working smoke)
            predictor.set_image(tile_path)            # ENCODE this piece once; conf sweep below reuses it
            for phrase in args.phrases:
                for c in args.conf:
                    predictor.args.conf = float(c)    # sweep conf without reloading the 3.45GB model
                    t0 = time.perf_counter()
                    res = predictor(text=[phrase])
                    dt = time.perf_counter() - t0
                    masks, boxes, conf = result_masks(res)
                    rec = {"dataset": it["dataset"], "plot": it["plot"], "stem": stem, "image": img_path,
                           "mode": tag, "phrase": phrase, "conf_thr": round(c, 2),
                           "num_instances": int(len(masks)),
                           "conf_mean": float(conf.mean()) if len(conf) else None, "seconds": round(dt, 2)}
                    summary.append(rec)
                    ph = phrase.replace(" ", "_")
                    lbl = f"{tag} {phrase} c{c:.2f}: {len(masks)}"
                    # split full-frame vs tiled into their own subfolders so the two --mode phases don't
                    # intermix; within each, filenames sort as full__c10..c50 / t00__c10..c50 (conf sweep)
                    bucket = "full_frame" if tag == "full" else "tiled"
                    ph_dir = os.path.join(out_dir, ph, bucket); os.makedirs(ph_dir, exist_ok=True)
                    out_png = os.path.join(ph_dir, f"{tag}__c{int(c * 100):02d}.jpg")
                    cv2.imwrite(out_png, make_overlay(tile_bgr, masks, boxes, lbl))
                    print(f"    {tag:5s} | {phrase:12s} c{c:.2f} -> {rec['num_instances']:4d} inst | {dt:.2f}s")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)            # drop the encode-input scratch dir
    # write a PER-MODE summary so a later phase (e.g. --mode tiles) does NOT overwrite an earlier one
    # (--mode full). summary.json stays as a back-compat copy of THIS run.
    for name in (f"summary_{args.mode}.json", "summary.json"):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(summary, f, indent=2)
    # quick flat-conf sanity: if instance count never changes across the conf sweep, PCS ignored args.conf
    by_key = {}
    for r in summary:
        by_key.setdefault((r["stem"], r["mode"], r["phrase"]), set()).add(r["num_instances"])
    flat = sum(1 for v in by_key.values() if len(v) == 1)
    if by_key and flat == len(by_key):
        print("\n⚠ instance count is FLAT across every conf sweep — SAM3SemanticPredictor likely ignored "
              "args.conf. Rebuild the predictor per conf to sweep it for real.")
    print(f"\nOverlays under {args.out}/<dataset>/<plot>/<stem>/  + summary.json  "
          f"(compare c10..c50 within a folder; full vs tXY across --mode runs)")


if __name__ == "__main__":
    main()
