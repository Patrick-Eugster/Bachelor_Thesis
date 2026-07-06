#!/usr/bin/env python3
"""Package a session's seed boxes into a CVAT-ready 'YOLO 1.1' import zip.

Reads the seed <stem>.txt files from input_plots/phone/<session>/gt_labeling/ and writes
a zip in the structure CVAT expects for annotation upload:

    obj.names                       (class names: wheat_head)
    obj.data                        (classes/names/train pointers)
    train.txt                       (one image path per line, basename must match the task frames)
    obj_train_data/<stem>.txt       (the YOLO boxes)

Workflow: create a CVAT task, upload gt_labeling/<stem>.jpg images, then upload this zip via
Actions -> Upload annotations -> format "YOLO 1.1". CVAT maps the .txt to frames by basename.

Usage:
    python src/analysis/make_cvat_seed_zip.py --session field_A/20250715
"""
import argparse
import glob
import os
import zipfile

PHONE_ROOT = "/workspace/input_plots/phone"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="e.g. field_A/20250715")
    ap.add_argument("--out", default=None, help="output zip path (default: session root)")
    args = ap.parse_args()

    gt_dir = os.path.join(PHONE_ROOT, args.session, "gt_labeling")
    # seed box files = IMG_*.txt, excluding the class-list files
    txts = sorted(p for p in glob.glob(os.path.join(gt_dir, "*.txt"))
                  if os.path.basename(p) not in ("labels.txt", "classes.txt"))
    if not txts:
        raise SystemExit(f"no seed .txt files in {gt_dir} (run make_gt_box_seeds.py first)")

    stems = [os.path.splitext(os.path.basename(p))[0] for p in txts]
    out = args.out or os.path.join(PHONE_ROOT, args.session,
                                   f"cvat_seed_{args.session.replace('/', '_')}.zip")

    obj_data = "classes = 1\nnames = data/obj.names\ntrain = data/train.txt\nbackup = backup/\n"
    train_txt = "".join(f"data/obj_train_data/{s}.jpg\n" for s in stems)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "wheat_head\n")
        z.writestr("obj.data", obj_data)
        z.writestr("train.txt", train_txt)
        for p, s in zip(txts, stems):
            with open(p) as f:
                z.writestr(f"obj_train_data/{s}.txt", f.read())

    n_boxes = sum(sum(1 for _ in open(p)) for p in txts)
    print(f"wrote {out}")
    print(f"  {len(stems)} images, {n_boxes} seed boxes:")
    for s in stems:
        print(f"    {s}")


if __name__ == "__main__":
    main()
