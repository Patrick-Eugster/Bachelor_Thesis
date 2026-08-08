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
import re
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


def agisoft_naming_guard(registered, ref_base):
    """Verify Agisoft's `<base>_<N>` naming really maps back to our base names — this is what makes the
    suffix-robust pin safe. Runs ONLY when the reconstruction's names carry a trailing _<N> (Agisoft
    ingestion index); our COLMAP / FIP naming returns None (guard N/A). For each suffixed name it checks
    the base is a real session image (bases_all_present — the hard requirement for pin correctness) and,
    as extra evidence, that _<N> equals the base's 0-based sorted index (suffix_is_sorted_index — a
    sanity signal, not required for the pin). Cheap: set/index only, no image decode."""
    suffixed = [n for n in registered if split_utils._norm_stem(n) != split_utils._stem(n)]
    if not suffixed:
        return None
    if not ref_base:
        return {"n_suffixed": len(suffixed), "ok": None, "note": "no base reference (input_uniform) to verify against"}
    ref_sorted = sorted(ref_base)
    idx = {n: i for i, n in enumerate(ref_sorted)}
    bad_base, bad_idx = [], []
    for n in registered:
        base = split_utils._norm_stem(n)
        if base not in idx:
            bad_base.append(n)
            continue
        m = re.match(r'^.*_(\d+)$', n)
        if m and int(m.group(1)) != idx[base]:
            bad_idx.append(f"{n}(N={m.group(1)},idx={idx[base]})")
    return {
        "n_suffixed": len(suffixed),
        "n_registered": len(registered),
        "bases_all_present": not bad_base,
        "suffix_is_sorted_index": not bad_idx,
        "bad_base": bad_base[:10],
        "bad_idx": bad_idx[:10],
        "ok": not bad_base,   # pin-safety hinges on bases being real; index equality is informational
    }


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
    pin_applies = split_utils.pin_applies(registered, pin)   # suffix-robust → also matches Agisoft _<N>
    train, test = split_utils.compute_eval_split(registered, pin_test=(pin if pin_applies else None))
    method = split_utils.split_method_label(registered, pin if pin_applies else None)

    # which intended inputs failed to register (drift out of the split). Only meaningful when the
    # input naming matches the reconstruction — Agisoft renames images, so a 0-overlap means
    # "different naming", not "all missing". Suppress the misleading list in that case.
    inputs, used_dir = _input_stems(os.path.join(source_path, cfg.image_subdir),
                                    os.path.join(recon_dir, "images"))
    names_match_inputs = bool(inputs & reg_set)
    not_registered = sorted(inputs - reg_set) if (inputs and names_match_inputs) else []

    # which pinned test views are missing from this reconstruction (the comparability killer).
    # suffix-robust so Agisoft's _<N> names don't read as "all missing".
    pin_missing = split_utils.pin_missing(registered, pin) if pin else []

    # Agisoft naming guard: when names are _<N>-suffixed, prove they map back to our base names before
    # we trust the suffix-robust pin. ref = the base (un-suffixed) input set (input_uniform).
    guard = agisoft_naming_guard(registered, inputs)

    # outcome: pass / fail (real drift) / n/a (pin exists but its naming doesn't apply here)
    if pin is None:
        result, exit_code = "pass", 0           # no pin to satisfy → fallback split, fine
    elif not pin_applies:
        result, exit_code = "n/a", 2            # e.g. Agisoft naming → can't compare to this pin
    elif pin_missing:
        result, exit_code = "fail", 1           # pin applies but test views dropped → DRIFT
    else:
        result, exit_code = "pass", 0
    # A failed naming guard invalidates the suffix-robust pin → hard fail regardless of the above.
    if guard and guard.get("ok") is False:
        result, exit_code = "fail", 1

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
    if guard is not None:
        if guard.get("ok") is None:
            print(f"{'Agisoft name guard:':<24} SKIPPED — {guard.get('note')}")
        elif guard["ok"]:
            idxnote = "base+sorted-index ✓" if guard["suffix_is_sorted_index"] else \
                      "bases ✓, but _<N> ≠ sorted index (ok — pin matches by base name)"
            print(f"{'Agisoft name guard:':<24} PASS — {guard['n_suffixed']} _<N> names map to base ({idxnote})")
        else:
            print(f"{'Agisoft name guard:':<24} FAIL — {len(guard['bad_base'])} names have NO base in the session: {guard['bad_base']}")
    print("-" * 60)
    print(f"{'Split method:':<24} {method}")
    print(f"{'Train / Test:':<24} {len(train)} / {len(test)}")
    print(f"{'Test views:':<24} {test}")
    print("-" * 60)
    if result == "fail" and guard and guard.get("ok") is False:
        fail_msg = "FAIL ✗ (Agisoft naming guard: names don't map to session base images)"
    else:
        fail_msg = "FAIL ✗ (pinned test views dropped — split drifted)"
    result_msg = {"pass": "PASS ✓", "fail": fail_msg,
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
        "naming_guard": guard,
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
