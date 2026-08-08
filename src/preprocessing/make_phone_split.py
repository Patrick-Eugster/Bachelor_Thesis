"""Write phone_split.json — the canonical train/test split for a phone session.

Phone has no transforms.json (FIP's split file), and the default llffhold-8 split is POSITIONAL:
every 8th sorted image. So two methods that register a different image subset end up testing on
different views → their PSNR/SSIM/LPIPS aren't comparable. This script fixes that by pinning the
split BY NAME, derived from the full intended image set (input_uniform/) before COLMAP drops anything.
Once phone_split.json exists at the session root, dataset_readers.py honors it for every method
(COLMAP, sparse_metric, ...), so they all hold out the same physical views.

Read-only except for the one JSON it writes. Run once per session.
"""
import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from wheat_utils import split_utils

IMG_EXTS = (".jpg", ".jpeg", ".png")


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/make_phone_split")
def main(cfg: DictConfig):
    """Build the canonical split from {source}/{image_subdir} names and write phone_split.json."""
    print("--- make_phone_split config ---")
    print(OmegaConf.to_yaml(cfg))
    print("-------------------------------")

    source_path = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot))
    img_dir = os.path.join(source_path, cfg.image_subdir)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"image dir not found: {img_dir} (run preprocess_uniform_size.py first?)")

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
    if not files:
        raise SystemExit(f"no images in {img_dir}")
    names = sorted(split_utils._stem(f) for f in files)
    name_set = set(names)

    # No pin here — we are CREATING the pin. compute_eval_split picks FIP cam-index if these happen to
    # be FIP-named, else llffhold (the phone case). Same function training uses → guaranteed identical.
    train, test = split_utils.compute_eval_split(names, pin_test=None, llffhold=cfg.llffhold)

    # Force every GT-labeled image into the test set — eval_2d compares a RENDERED (held-out) view
    # against its manual GT mask, so a GT image that landed in train would be unusable for evaluation.
    # GT stems come from manual_label/<stem><gt_mask_suffix>.png.
    gt_dir = os.path.join(source_path, cfg.gt_label_subdir)
    gt_stems, gt_missing = [], []
    if os.path.isdir(gt_dir):
        suf = cfg.gt_mask_suffix + ".png"
        for f in sorted(os.listdir(gt_dir)):
            if f.endswith(suf):
                stem = f[: -len(suf)]
                (gt_stems if stem in name_set else gt_missing).append(stem)
    forced = [g for g in gt_stems if g not in set(test)]   # GT that llffhold had put in train
    test = sorted(set(test) | set(gt_stems))
    train = [n for n in names if n not in set(test)]
    method = split_utils.split_method_label(names, None)
    if gt_stems:
        method += f" + {len(gt_stems)} GT forced to test"

    out_path = os.path.join(source_path, cfg.output)
    if os.path.exists(out_path) and not cfg.overwrite:
        raise SystemExit(f"{out_path} already exists — pass overwrite=true to replace it "
                         f"(protects a split you may already be comparing against).")

    payload = {
        "field": str(cfg.field),
        "plot": str(cfg.plot),
        "image_subdir": cfg.image_subdir,
        "split_method": method,
        "llffhold": cfg.llffhold,
        "n_images": len(names),
        "gt_views": sorted(gt_stems),        # GT-labeled stems (all guaranteed to be in test_views)
        "train_views": train,
        "test_views": test,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 56)
    print("      PHONE SPLIT WRITTEN")
    print("=" * 56)
    print(f"{'Session:':<22} {cfg.field}/{cfg.plot}")
    print(f"{'Source images:':<22} {len(names)}  (from {cfg.image_subdir}/)")
    print(f"{'Split method:':<22} {method}")
    print(f"{'Train / Test:':<22} {len(train)} / {len(test)}")
    if gt_stems:
        print(f"{'GT views (in test):':<22} {sorted(gt_stems)}")
        print(f"{'GT forced from train:':<22} {forced if forced else 'none (already in test)'}")
    if gt_missing:
        print(f"{'⚠ GT not in image set:':<22} {gt_missing}  (stem mismatch — check naming)")
    print(f"{'Test views:':<22} {test}")
    print(f"{'Written to:':<22} {out_path}")
    print("=" * 56 + "\n")
    return payload


if __name__ == "__main__":
    main()
