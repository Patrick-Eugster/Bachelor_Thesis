# Documentation

Setup guides and reference docs for the Wheat3DGS project.

## Files in this folder

| File | Tracked by git | Purpose |
|---|---|---|
| [`INSTALL_COLMAP_CUDA.md`](INSTALL_COLMAP_CUDA.md) | ✅ yes | Step-by-step guide for building COLMAP from source with CUDA support. Use when setting up a new Docker container or machine. Covers all dependencies, the cmake config, GPU compute-capability reference, and troubleshooting for every error we hit during the build. |
| `euler_setup.md` | ❌ gitignored | Personal notes on running this codebase on the ETH Euler HPC cluster (code transfer, CUDA submodule compilation, SLURM job script, rsync workflow). Author-specific — not committed. |
| `CHANGES.md` | ❌ gitignored | Personal per-file change log (what was changed and why) — author's dev notes, not project history. |

## Where other docs live

- **Project-wide guidance for Claude Code:** [`../CLAUDE.md`](../CLAUDE.md) (workspace root)
- **Per-module READMEs:** under `src/<module>/README.md`
  - [`src/preprocessing/README.md`](../src/preprocessing/README.md) — phone preprocessing (uniform-size + COLMAP), full empirical test log
  - [`src/reconstruction/README.md`](../src/reconstruction/README.md) — 3DGS training, rendering, metrics
  - [`src/segmentation_3d/README.md`](../src/segmentation_3d/README.md) — FlashSplat 3D segmentation
  - [`src/mask_generation/yolo_sam_v1/README.md`](../src/mask_generation/yolo_sam_v1/README.md) — YOLO+SAM pipelining details
  - [`src/mask_generation/metrics/README.md`](../src/mask_generation/metrics/README.md) — YOLO evaluation metrics
