"""SAM3 text-prompt (Promptable Concept Segmentation) smoke test.

Runs SAM3's TEXT prompts over a few images with several concept phrases and dumps
an overlay per (image, phrase) plus a JSON summary (instance count + confidence + time).
This is the "which phrase fires on wheat heads?" check BEFORE wiring text mode into the
pipeline — it uses ultralytics' SAM3SemanticPredictor directly, no YOLO boxes involved.

Runs on Euler (H100/A100) — SAM3's encoder is ~26 GB in FP32 (fits H100 80 GB); pass
--fp32 to force full precision, otherwise it uses quantize=16 (FP16, ~13 GB) so it also
fits a 24 GB card. NOT runnable on the 16 GB local WSL GPU.

Example (on the code-server):
    python src/analysis/sam3_text_smoke.py \
        --images input_plots/fip/plot_461/images/FPWW036_SR0461_1_FIP2_cam_01.png \
                 input_plots/phone/field_A/20250627/images/IMG_20250627_100946.jpg \
        --phrases "wheat head" "wheat spike" "wheat ear" "wheat" \
        --out docs/analysis_results/sam3_text_smoke
"""
import argparse
import json
import os
import time

import cv2
import numpy as np


DEFAULT_IMAGES = [
    "input_plots/fip/plot_461/images/FPWW036_SR0461_1_FIP2_cam_01.png",
    "input_plots/phone/field_A/20250627/images/IMG_20250627_100946.jpg",
]
DEFAULT_PHRASES = ["wheat head", "wheat spike", "wheat ear", "wheat"]


def build_predictor(weight, fp32, conf, imgsz):
    """Build a SAM3SemanticPredictor (the TEXT-prompt / concept interface).
    quantize=16 → FP16 (default, halves VRAM); fp32 forces full precision."""
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = {
        "model": weight,
        "task": "segment",
        "mode": "predict",
        "conf": conf,
        "save": False,
        "verbose": False,
    }
    if imgsz:
        overrides["imgsz"] = imgsz
    if not fp32:
        overrides["quantize"] = 16  # FP16
    return SAM3SemanticPredictor(overrides=overrides)


def result_masks(res):
    """Pull (masks NxHxW uint8 0/1, boxes Nx4, conf N) out of one ultralytics Result,
    tolerating the case where masks or boxes are missing."""
    masks = np.zeros((0, 0, 0), dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)
    conf = np.zeros((0,), dtype=np.float32)
    if getattr(res, "masks", None) is not None and res.masks.data is not None and len(res.masks.data):
        masks = (res.masks.data.detach().cpu().numpy() > 0.5).astype(np.uint8)
    if getattr(res, "boxes", None) is not None and len(res.boxes):
        boxes = res.boxes.xyxy.detach().cpu().numpy()
        conf = res.boxes.conf.detach().cpu().numpy()
    return masks, boxes, conf


def make_overlay(img_bgr, masks, boxes, phrase):
    """Blend all instance masks (one translucent green union) + draw boxes + a count label.
    Just for eyeballing which phrase actually lands on heads — not a per-instance colour map."""
    vis = img_bgr.copy()
    if len(masks):
        H, W = img_bgr.shape[:2]
        union = np.zeros((H, W), dtype=bool)
        for m in masks:
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            union |= m.astype(bool)
        green = np.zeros_like(vis)
        green[union] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 1.0, green, 0.45, 0)
    for b in boxes.astype(int):
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    cv2.putText(vis, f"{phrase}: {len(masks)} inst", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    ap.add_argument("--phrases", nargs="+", default=DEFAULT_PHRASES)
    ap.add_argument("--weight", default="src/mask_generation/weights/sam3.pt")
    ap.add_argument("--out", default="docs/analysis_results/sam3_text_smoke")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=0, help="0 = model default (~1036)")
    ap.add_argument("--fp32", action="store_true", help="force FP32 (needs ~26 GB VRAM)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    predictor = build_predictor(args.weight, args.fp32, args.conf, args.imgsz)

    summary = []
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"[skip] missing {img_path}")
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        # set_image ENCODES ONCE; every phrase below reuses those features (cheap).
        predictor.set_image(img_path)
        for phrase in args.phrases:
            t0 = time.perf_counter()
            res = predictor(text=[phrase])
            dt = time.perf_counter() - t0
            r = res[0] if isinstance(res, (list, tuple)) else res
            masks, boxes, conf = result_masks(r)
            rec = {
                "image": img_path,
                "phrase": phrase,
                "num_instances": int(len(masks)),
                "conf_min": float(conf.min()) if len(conf) else None,
                "conf_mean": float(conf.mean()) if len(conf) else None,
                "conf_max": float(conf.max()) if len(conf) else None,
                "seconds": round(dt, 2),
            }
            summary.append(rec)
            print(f"  {stem:32s} | {phrase:12s} -> {rec['num_instances']:4d} inst "
                  f"| conf~{rec['conf_mean']} | {dt:.2f}s")
            out_png = os.path.join(args.out, f"{stem}__{phrase.replace(' ', '_')}.jpg")
            cv2.imwrite(out_png, make_overlay(img_bgr, masks, boxes, phrase))

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOverlays + summary.json -> {args.out}")


if __name__ == "__main__":
    main()
