"""Re-color a segmentation PLY with a HIGH-CONTRAST palette so adjacent wheat heads get visibly
different colors, instead of the default export_colored_ply.py ramp hue=(head_id-1)/n_heads (which
makes neighbouring IDs near-identical shades when there are ~2000 heads).

Why: with a thin ramp a block of one apparent colour can be many distinct heads OR one over-merged
head — you can't tell them apart by eye. Here each head's hue hops by the golden-ratio conjugate
(0.618) so consecutive IDs land far apart on the colour wheel, and saturation/brightness alternate in
a short cycle for extra separation. Then in supersplat.dev a patch that STAYS one colour is genuinely
one (merged) head, while separate heads break into clearly different colours — directly answering
"are multiple heads sharing an ID here?".

Reuses the exact load/save + SH-DC colour maths of src/segmentation_3d/export_colored_ply.py; only the
per-head colour picker changed. Writes a NEW file (default gaussians_colored_contrast.ply) — never
overwrites the original gaussians_colored.ply. Local, CPU/GPU-light, no re-seg, no render.
"""
import os
import argparse
import colorsys
import torch

from gaussians.scene.gaussian_model import GaussianModel

SH_C0 = 0.28209479177387814          # zeroth-order SH coefficient (same constant as the export script)
GOLDEN = 0.6180339887498949          # golden-ratio conjugate — max-spread hue hop per head
# short cycles so neighbouring IDs also differ in S and V, not just hue
SAT_CYCLE = [0.95, 0.70, 0.85, 0.60]
VAL_CYCLE = [0.95, 0.75]


def contrast_color(head_id):
    """Golden-ratio hue hop + alternating sat/val so consecutive head IDs look maximally different."""
    hue = (head_id * GOLDEN) % 1.0
    sat = SAT_CYCLE[head_id % len(SAT_CYCLE)]
    val = VAL_CYCLE[head_id % len(VAL_CYCLE)]
    return colorsys.hsv_to_rgb(hue, sat, val)


def main():
    ap = argparse.ArgumentParser(description="Re-color a seg PLY with a high-contrast per-head palette")
    ap.add_argument("--gaussians_ply", required=True, help="gaussians.ply from step 4 (the uncolored model)")
    ap.add_argument("--labels_path", required=True, help="all_obj_labels.pth from step 4")
    ap.add_argument("--output_ply", default=None, help="output (default: gaussians_colored_contrast.ply next to labels)")
    ap.add_argument("--sh_degree", type=int, default=3, help="SH degree used in training (must match, default 3)")
    a = ap.parse_args()

    out = a.output_ply or os.path.join(os.path.dirname(os.path.abspath(a.labels_path)),
                                       "gaussians_colored_contrast.ply")
    if os.path.exists(out):
        raise SystemExit(f"ABORT: {out} already exists — refusing to overwrite. Pass a different --output_ply.")

    print(f"Loading gaussians from {a.gaussians_ply}...")
    gs = GaussianModel(sh_degree=a.sh_degree)
    gs.load_ply(a.gaussians_ply)
    print(f"Loaded {len(gs.get_xyz)} Gaussians")

    print(f"Loading labels from {a.labels_path}...")
    all_obj_labels = torch.load(a.labels_path)          # (n_heads+1, G) bool; row 0 = background
    n_heads = all_obj_labels.shape[0] - 1
    print(f"Baking HIGH-CONTRAST colors for {n_heads} heads...")

    n_labeled = 0
    for head_id in range(1, n_heads + 1):
        mask = all_obj_labels[head_id].bool()
        if not mask.is_cuda:
            mask = mask.cuda()
        count = mask.sum().item()
        if count == 0:
            continue
        n_labeled += count
        r, g, b = contrast_color(head_id)
        color = torch.tensor([(r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0],
                             dtype=torch.float32, device="cuda")
        gs._features_dc.data[mask] = color.view(1, 1, 3)
        if gs._features_rest is not None and gs._features_rest.shape[1] > 0:
            gs._features_rest.data[mask] = 0.0

    total = len(gs.get_xyz)
    print(f"Colored {n_labeled}/{total} Gaussians ({100 * n_labeled / total:.1f}%)")
    print(f"Saving to {out}...")
    gs.save_ply(out)
    print(f"Done. Drag {out} into supersplat.dev — a patch that stays ONE colour is one (merged) head; "
          f"separate heads now show clearly different colours.")


if __name__ == "__main__":
    main()
