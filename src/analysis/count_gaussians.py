"""Count the number of Gaussians in trained 3DGS models by reading the vertex
count straight from each point_cloud.ply header (no full load, so it is fast and
uses no VRAM). Builds two tables: the phone densification 2x2 (baseline / absgrad /
dense17k_noabsgrad / dense17k) averaged over the four evaluated sessions, and the
FIP AbsGS 15k-vs-30k training-length pair. Missing local plys (e.g. phone models
whose point_cloud/ was not pulled from Euler) are reported as such.

Run: python src/analysis/count_gaussians.py

Densification threshold (verified from the job scripts, not the per-run config.yaml
which is stale for the dense17k runs): the threshold moves with the AbsGS flag, so
BOTH the FIP and the phone AbsGS runs use densify_grad_threshold=0.0008, and every
default (non-AbsGS) run uses 0.0002. The FIP-vs-phone count divergence (AbsGS gives
fewer Gaussians on FIP but more on phone) is therefore not a threshold difference.
"""
import glob
import os

RESULTS = "results/reconstruction"

# the four phone sessions used in the reconstruction evaluation
PHONE_SESSIONS = [
    "phone/field_A/20250627",
    "phone/field_A/20250715",
    "phone/field_D/20250627",
    "phone/field_D/20250715",
]

# maps a Table 7.21 setting to (experiment folder, iteration)
PHONE_DENSIFY = {
    "15k, default":  ("baseline",           15000),
    "15k, AbsGS":    ("absgrad",            15000),
    "30k, default":  ("dense17k_noabsgrad", 30000),
    "30k, AbsGS":    ("dense17k",           30000),
}


def ply_vertex_count(path):
    """Reads just the ascii header of a binary ply and returns its vertex count,
    or None if the file is missing."""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        head = f.read(2048).decode("latin-1", errors="ignore")
    for line in head.splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[-1])
    return None


def phone_ply(session, exp, iteration):
    """Builds the expected ply path for a phone densification run."""
    return os.path.join(
        RESULTS, session, "opencv", "vanilla_3dgs", exp,
        "point_cloud", f"iteration_{iteration}", "point_cloud.ply",
    )


# Phone densification models keep only baseline/ locally; the absgrad and 30k
# (dense17k*) point clouds live on Euler (point_cloud/ is excluded from the pull).
# These counts were read from the Euler login node with:
#   for s in field_A/20250627 field_A/20250715 field_D/20250627 field_D/20250715; do
#     for v in baseline:15000 absgrad:15000 dense17k_noabsgrad:30000 dense17k:30000; do
#       e=${v%:*}; it=${v#*:}
#       f=results/reconstruction/phone/$s/opencv/vanilla_3dgs/$e/point_cloud/iteration_$it/point_cloud.ply
#       head -c 2048 "$f" | grep -a 'element vertex'
#     done
#   done
# session order: A/0627, A/0715, D/0627, D/0715
PHONE_COUNTS_EULER = {
    "15k, default": [1224120, 1554175, 1028951, 1263497],
    "15k, AbsGS":   [4712226, 2833669, 3718979, 3921981],
    "30k, default": [2349155, 2218285, 1974195, 2208918],
    "30k, AbsGS":   [8721639, 4385736, 4828730, 4688000],
}


def phone_table_euler():
    """Prints the full phone 2x2 from the Euler-read header counts (recorded above),
    with per-session values, the average, and the multiple of the 15k-default baseline."""
    print("== Phone densification (OPENCV), Gaussian count [Euler headers] ==")
    print("setting".ljust(16) + "A/0627 A/0715 D/0627 D/0715".rjust(0))
    base = sum(PHONE_COUNTS_EULER["15k, default"]) / 4
    for setting, counts in PHONE_COUNTS_EULER.items():
        avg = sum(counts) / len(counts)
        cells = "".join(f"{c:,}".rjust(12) for c in counts)
        print(setting.ljust(16) + cells + f"{int(avg):,}".rjust(14) + f"{avg/base:.1f}x".rjust(8))
    print()


def phone_table():
    """Prints per-session and averaged Gaussian counts for the phone 2x2."""
    print("== Phone densification (OPENCV), Gaussian count [local plys only] ==")
    header = "setting".ljust(16) + "".join(s.split("/")[-1].rjust(12) for s in PHONE_SESSIONS) + "avg".rjust(14)
    print(header)
    for setting, (exp, it) in PHONE_DENSIFY.items():
        counts = [ply_vertex_count(phone_ply(s, exp, it)) for s in PHONE_SESSIONS]
        cells = "".join(("--" if c is None else f"{c:,}").rjust(12) for c in counts)
        present = [c for c in counts if c is not None]
        avg = f"{sum(present)//len(present):,}" if present else "--"
        print(setting.ljust(16) + cells + avg.rjust(14))
    print()


def fip_absgrad_onoff():
    """Prints FIP Gaussian count with AbsGS off (default densification, gsplat) vs
    on (AbsGS), per plot at 15k. The count is frozen after densify_until (~11k), so
    15k and 30k are identical on FIP and we report 15k."""
    print("== FIP, AbsGS off vs on, Gaussian count (gsplat, 15k) ==")
    print("plot".ljust(8) + "AbsGS off".rjust(14) + "AbsGS on".rjust(14) + "ratio".rjust(8))
    offs, ons = [], []
    for plot_dir in sorted(glob.glob(os.path.join(RESULTS, "fip", "plot_*"))):
        plot = os.path.basename(plot_dir)
        off = ply_vertex_count(os.path.join(
            plot_dir, "vanilla_3dgs", "test_gsplat_full",
            "point_cloud", "iteration_15000", "point_cloud.ply"))
        on = None
        for exp in ("test_absgrad", "test_absgrad_v2"):  # 461 vs 462-467
            on = ply_vertex_count(os.path.join(
                plot_dir, "vanilla_3dgs", exp,
                "point_cloud", "iteration_15000", "point_cloud.ply"))
            if on is not None:
                break
        if off is None and on is None:
            continue
        ratio = f"{on/off:.2f}x" if (off and on) else "--"
        print(plot.ljust(8)
              + ("--" if off is None else f"{off:,}").rjust(14)
              + ("--" if on is None else f"{on:,}").rjust(14)
              + ratio.rjust(8))
        if off:
            offs.append(off)
        if on:
            ons.append(on)
    if offs and ons:
        ao, an = sum(offs) // len(offs), sum(ons) // len(ons)
        print("AVG".ljust(8) + f"{ao:,}".rjust(14) + f"{an:,}".rjust(14) + f"{an/ao:.2f}x".rjust(8))
    print()


if __name__ == "__main__":
    phone_table_euler()
    fip_absgrad_onoff()
