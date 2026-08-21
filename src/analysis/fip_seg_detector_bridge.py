"""Builds the FIP detector->3D-seg 'bridge' table for the fipseg15k_pp experiment: does 2D mask
quality carry over to the 3D segmentation? Reads each variant's eval_2d/metrics_2d.json (binary
2D seg eval on the single labeled GT camera per plot) for the three detector-mask inputs
(YOLOv5 640, YOLOv5 1280, YOLO11) across plots 461-467, prints the per-plot table plus the
per-detector mean over the seven plots. Run from repo root."""
import json
import statistics as st

PLOTS = range(461, 468)
DETS = ["seg_yv5_640", "seg_yv5_1280", "seg_yolo11"]
BASE = "results/reconstruction/fip/plot_{p}/vanilla_3dgs/fipseg15k_pp/segmentation_3d/{d}/eval_2d/metrics_2d.json"


def main():
    rows = []
    for p in PLOTS:
        for d in DETS:
            m = json.load(open(BASE.format(p=p, d=d)))[0]
            rows.append((p, d, m["iou"], m["precision"], m["recall"], m["f1"],
                         m["gt_head_count"], m["pred_head_count"], m["count_error_ratio"]))

    hdr = ("plot", "detector", "IoU", "P", "R", "F1", "gtHd", "prHd", "cntErr")
    print("{:>5} {:>14} {:>6} {:>6} {:>6} {:>6} {:>5} {:>5} {:>7}".format(*hdr))
    for r in rows:
        print("{:>5} {:>14} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>5} {:>5} {:>7.3f}".format(*r))

    print("\n=== per-detector MEAN over 7 plots ===")
    print("{:>14} {:>6} {:>6} {:>6} {:>6} {:>7}".format("detector", "IoU", "P", "R", "F1", "cntErr"))
    for d in DETS:
        sub = [r for r in rows if r[1] == d]
        print("{:>14} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>7.3f}".format(
            d,
            st.mean(x[2] for x in sub), st.mean(x[3] for x in sub),
            st.mean(x[4] for x in sub), st.mean(x[5] for x in sub),
            st.mean(x[8] for x in sub)))


if __name__ == "__main__":
    main()
