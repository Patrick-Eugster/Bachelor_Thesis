"""Tiny shared helper for the `only_labeled_images` filter.

Ground truth comes in two flavours in `manual_label/`:
  - BOX GT   : `<stem>.txt`           (YOLO boxes, for eval_yolo_boxes)
  - MASK GT  : `<stem>_sets/` folder  (the point-GT tool's instance masks) + `<stem>_gt_mask.png`

Originally the pipelines only recognised `.txt`, so on a mask-GT-only phone session
`only_labeled_images=true` matched nothing. This makes it recognise both, so the flag targets exactly
the GT image(s) of a session for either kind of ground truth.
"""

import os


def gt_labeled_stems(label_dir):
    """Return the set of image stems that have ANY ground truth in label_dir — a `.txt` box file, a
    `<stem>_sets/` mask-GT folder, or a `<stem>_gt_mask.png`. Empty set if the dir doesn't exist."""
    stems = set()
    if not os.path.isdir(label_dir):
        return stems
    for f in os.listdir(label_dir):
        if f.endswith(".txt"):
            stems.add(f[:-4])
        elif f.endswith("_gt_mask.png"):
            stems.add(f[: -len("_gt_mask.png")])
        elif f.endswith("_sets"):                      # the authoritative mask-GT marker
            stems.add(f[: -len("_sets")])
    return stems
