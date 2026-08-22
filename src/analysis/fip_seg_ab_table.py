"""Print the FIP 3D-seg A/B tables (principal-point on/off, AbsGrad on/off) from the
seg_yv5_1280 eval_2d metrics, so the numbers are reproducible instead of one-off.

All three experiments use the SAME yv5_1280 masks; control = fipseg15k_pp (pp-on, absgrad-off):
  fipseg15k_pp      = pp ON,  absgrad OFF   (control for both A/Bs)
  fipseg15k_ppoff   = pp OFF, absgrad OFF   (pp A/B)
  fipseg15k_absgrad = pp ON,  absgrad ON    (absgrad A/B)
Read-only; writes nothing.
"""
import json
import os

ROOT = "results/reconstruction/fip"
PLOTS = [f"plot_46{i}" for i in range(1, 8)]
EXPS = {"pp_on": "fipseg15k_pp", "pp_off": "fipseg15k_ppoff", "absgrad": "fipseg15k_absgrad"}


def val(exp, plot, key):
    """one metric from a plot's seg_yv5_1280 eval_2d (metrics_2d.json is a 1-element list)."""
    f = os.path.join(ROOT, plot, "vanilla_3dgs", exp, "segmentation_3d", "seg_yv5_1280",
                     "eval_2d", "metrics_2d.json")
    return json.load(open(f))[0][key]


def main():
    """print IoU/F1/precision/recall tables + the pp and absgrad deltas."""
    for metric in ["iou", "f1", "precision", "recall"]:
        print(f"\n=== {metric.upper()} (seg_yv5_1280) ===")
        print(f"{'plot':9}{'pp_on':>9}{'pp_off':>9}{'absgrad':>9}")
        acc = {k: [] for k in EXPS}
        for p in PLOTS:
            row = f"{p:9}"
            for k, e in EXPS.items():
                v = val(e, p, metric)
                acc[k].append(v)
                row += f"{v:>9.3f}"
            print(row)
        means = {k: sum(acc[k]) / len(PLOTS) for k in EXPS}
        print(f"{'MEAN':9}" + "".join(f"{means[k]:>9.3f}" for k in EXPS))
        if metric == "iou":
            print(f"  pp fix (on-off): {means['pp_on']-means['pp_off']:+.3f}   "
                  f"absgrad (on-control): {means['absgrad']-means['pp_on']:+.3f}")


if __name__ == "__main__":
    main()
