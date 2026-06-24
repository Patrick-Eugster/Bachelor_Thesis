"""Verify a reconstruction's 3DGS train/test split — and flag registration drift.

Why this exists: the held-out test set is what PSNR/SSIM/LPIPS are measured on. For FIP we want it
to match the paper (transforms.json); for phone we want it identical across methods (COLMAP,
sparse_metric, ...). Both are only guaranteed if every pinned test view actually registered. This
script reads a reconstruction's sparse/0, runs the SAME split logic training uses
(wheat_utils.split_utils — single source of truth), and reports:
  - how many input images failed to register (drift out of the split),
  - the pin source (transforms.json / phone_split.json) and whether every pinned test view registered,
  - the resulting train / test split.
Exits non-zero when pinned test views are missing, so it can double as a pre-comparison gate/test.

Read-only except for one logs/split_check.json. Works for FIP and phone.
"""
import json
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from wheat_utils import split_utils

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _input_stems(*dirs):
    """Return the set of image stems found in the first of `dirs` that exists (or empty set)."""
    for d in dirs:
        if d and os.path.isdir(d):
            return {split_utils._stem(f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)}, d
    return set(), None


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/check_split")
def main(cfg: DictConfig):
    """Read sparse/0, compute the split via the shared helper, and report MATCH / drift."""
    print("--- check_split config ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------")

    # FIP has no field level (input_plots/fip/plot_461); phone does (input_plots/phone/field/date).
    if str(cfg.dataset.name) == "fip":
        source_path = os.path.join(cfg.dataset.input_dir, str(cfg.plot))
    else:
        source_path = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot))
    recon_dir = os.path.join(source_path, cfg.sfm_subdir) if cfg.sfm_subdir else source_path

    # sparse model: prefer sparse/0, fall back to sparse/ (some FIP plots keep the model there).
    sparse0 = os.path.join(recon_dir, "sparse", "0")
    if not (os.path.isfile(os.path.join(sparse0, "images.txt")) or
            os.path.isfile(os.path.join(sparse0, "images.bin"))):
        sparse0 = os.path.join(recon_dir, "sparse")
    registered = split_utils.read_colmap_image_names(sparse0)
    reg_set = set(registered)

    # the pin training would use for THIS reconstruction (transforms.json / phone_split.json)
    pin = split_utils.load_pin_test(recon_dir)
    pin_applies = bool(pin and (pin & reg_set))   # different naming (Agisoft) → pin won't apply
    train, test = split_utils.compute_eval_split(registered, pin_test=(pin if pin_applies else None))
    method = split_utils.split_method_label(registered, pin if pin_applies else None)

    # which intended inputs failed to register (drift out of the split). Only meaningful when the
    # input naming matches the reconstruction — Agisoft renames images, so a 0-overlap means
    # "different naming", not "all missing". Suppress the misleading list in that case.
    inputs, used_dir = _input_stems(os.path.join(source_path, cfg.image_subdir),
                                    os.path.join(recon_dir, "images"))
    names_match_inputs = bool(inputs & reg_set)
    not_registered = sorted(inputs - reg_set) if (inputs and names_match_inputs) else []

    # which pinned test views are missing from this reconstruction (the comparability killer)
    pin_missing = sorted(pin - reg_set) if pin else []

    # outcome: pass / fail (real drift) / n/a (pin exists but its naming doesn't apply here)
    if pin is None:
        result, exit_code = "pass", 0           # no pin to satisfy → fallback split, fine
    elif not pin_applies:
        result, exit_code = "n/a", 2            # e.g. Agisoft naming → can't compare to this pin
    elif pin_missing:
        result, exit_code = "fail", 1           # pin applies but test views dropped → DRIFT
    else:
        result, exit_code = "pass", 0

    print("\n" + "=" * 60)
    print("      SPLIT CHECK")
    print("=" * 60)
    print(f"{'Reconstruction:':<24} {recon_dir}")
    print(f"{'Registered images:':<24} {len(registered)}")
    if used_dir and names_match_inputs:
        print(f"{'Intended inputs:':<24} {len(inputs)}  (from {os.path.relpath(used_dir, source_path)}/)")
        if not_registered:
            print(f"{'NOT registered:':<24} {len(not_registered)}  {not_registered}")
    print("-" * 60)
    if pin is None:
        print(f"{'Pin source:':<24} none → fallback rule")
    else:
        print(f"{'Pin source:':<24} transforms.json / phone_split.json ({len(pin)} test views)")
        if not pin_applies:
            print(f"{'Pin applies:':<24} NO — 0 names match (different naming, e.g. Agisoft); using fallback")
        elif pin_missing:
            print(f"{'Pinned test MISSING:':<24} {len(pin_missing)}  {pin_missing}   <-- DRIFT")
        else:
            print(f"{'Pinned test:':<24} all {len(pin)} registered ✓")
    print("-" * 60)
    print(f"{'Split method:':<24} {method}")
    print(f"{'Train / Test:':<24} {len(train)} / {len(test)}")
    print(f"{'Test views:':<24} {test}")
    print("-" * 60)
    result_msg = {"pass": "PASS ✓",
                  "fail": "FAIL ✗ (pinned test views dropped — split drifted)",
                  "n/a": "N/A — pin exists but doesn't apply to this naming (not comparable by name)"}[result]
    print(f"{'RESULT:':<24} {result_msg}")
    print("=" * 60 + "\n")

    log_dir = os.path.join(recon_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out = {
        "reconstruction": recon_dir,
        "n_registered": len(registered),
        "n_intended_inputs": len(inputs),
        "not_registered": not_registered,
        "pin_source": None if pin is None else "transforms.json/phone_split.json",
        "pin_n_test": 0 if pin is None else len(pin),
        "pin_applies": pin_applies,
        "pin_missing": pin_missing,
        "split_method": method,
        "n_train": len(train),
        "n_test": len(test),
        "test_views": test,
        "result": result,
    }
    with open(os.path.join(log_dir, "split_check.json"), "w") as f:
        json.dump(out, f, indent=2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
