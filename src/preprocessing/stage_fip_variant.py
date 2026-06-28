"""Stage one FIP reprocessed variant into the normal input_plots/fip/ layout.

The supervisor's FIP_single_row_exp_2024_reprocessed/ drop keeps each plot's data inside
demoanlage2025_v0/.../plot_<id>/colmap_reprocessed/<variant>/ (variant = undistorted_jpg |
undistorted_png | distorted_jpg | distorted_png), each already a ready images/ + sparse/0/. This copies
(or symlinks) a chosen plot+variant out into input_plots/fip/<plot>_<variant>/ so the rest of the
pipeline (run_reconstruction.py etc.) can consume it like any other FIP plot — without everything living
under demoanlage/. The per-plot marker_projections.csv is staged too (needed for the FIP metric
benchmark, exp #2).

Default is a real COPY (so input_plots/ is self-contained); pass --link for zero-cost symlinks.

Usage:
    # copy plot 461's undistorted-jpg variant -> input_plots/fip/plot_461_undistorted_jpg/
    python src/preprocessing/stage_fip_variant.py --plot 461 --variant undistorted_jpg
    # all 4 variants of one plot, as symlinks
    python src/preprocessing/stage_fip_variant.py --plot 461 --variant all --link
"""

import argparse
import os
import shutil

VARIANTS = ["undistorted_jpg", "undistorted_png", "distorted_jpg", "distorted_png"]
DEFAULT_SRC = "demoanlage2025_v0/FIP_single_row_exp_2024_reprocessed"
DEFAULT_DST = "input_plots/fip"


def _place(src, dst, link, overwrite):
    """Copy or symlink src -> dst (a dir or file). Skips/overwrites per flags."""
    if os.path.lexists(dst):
        if not overwrite:
            print(f"  exists, skipping (use --overwrite): {dst}")
            return
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        else:
            shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if link:
        os.symlink(os.path.abspath(src), dst)
    elif os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def stage(plot, variant, src_root, dst_root, link, overwrite):
    """Stage one plot+variant into input_plots/fip/<plot>_<variant>/ (images/ + sparse/ + marker csv)."""
    plot_name = plot if str(plot).startswith("plot_") else f"plot_{plot}"
    src_plot = os.path.join(src_root, plot_name)
    src_var = os.path.join(src_plot, "colmap_reprocessed", variant)
    if not os.path.isdir(src_var):
        print(f"  MISSING source variant: {src_var} — skipping")
        return
    dst = os.path.join(dst_root, f"{plot_name}_{variant}")
    print(f"[{plot_name}/{variant}] -> {dst}  ({'symlink' if link else 'copy'})")
    _place(os.path.join(src_var, "images"), os.path.join(dst, "images"), link, overwrite)
    _place(os.path.join(src_var, "sparse"), os.path.join(dst, "sparse"), link, overwrite)
    # marker pixels live at plot level (same for every variant) — stage a copy so the plot is self-contained
    mk = os.path.join(src_plot, "marker_projections.csv")
    if os.path.isfile(mk):
        _place(mk, os.path.join(dst, "marker_projections.csv"), link, overwrite)


def main():
    """CLI: stage FIP variant(s) into input_plots/fip/."""
    ap = argparse.ArgumentParser(description="Stage a FIP reprocessed variant into input_plots/fip/.")
    ap.add_argument("--plot", required=True, help="plot id, e.g. 461 or plot_461")
    ap.add_argument("--variant", default="undistorted_jpg",
                    help=f"one of {VARIANTS}, or 'all' for every variant")
    ap.add_argument("--src-root", default=DEFAULT_SRC, help="FIP reprocessed root")
    ap.add_argument("--dst-root", default=DEFAULT_DST, help="destination root (input_plots/fip)")
    ap.add_argument("--link", action="store_true", help="symlink instead of copy (zero disk cost)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing staged dir")
    args = ap.parse_args()

    variants = VARIANTS if args.variant == "all" else [args.variant]
    for v in variants:
        stage(args.plot, v, args.src_root, args.dst_root, args.link, args.overwrite)


if __name__ == "__main__":
    main()
