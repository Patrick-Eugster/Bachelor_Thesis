# Installation

> **Status:** this is a short placeholder. The full Docker-based install is being
> written and will land here together with the `Dockerfile`.

The recommended way to install Phone-Wheat3DGS is **with Docker**. The pipeline
combines several parts that are sensitive to exact CUDA and PyTorch versions — the
3DGS renderer and the FlashSplat segmentation are compiled CUDA extensions, and mask
generation needs a separate, newer PyTorch — so a plain local install tends to run
into version conflicts. A single Docker image bundles everything (PyTorch, the
compiled CUDA extensions, and a CUDA-enabled COLMAP) into one working environment,
which is why it is the main install path.

**Detailed Docker build and run instructions are coming soon**, along with the
`Dockerfile` in the repository root.

## Model weights

The YOLO and SAM checkpoints are not shipped with the repository (they are gitignored
under `src/mask_generation/weights/`). Download them into that folder before running
mask generation.

- Download link: _to be added._

## Manual conda setup

A manual, conda-based setup (the way the pipeline runs on our compute cluster) will
be documented here later as a secondary option.
