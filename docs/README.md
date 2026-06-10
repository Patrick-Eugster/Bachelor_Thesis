# Documentation Index

Quick map of every doc in this folder, grouped by topic. Each line is a one-sentence summary — open the file for the full write-up. Legend: ✅ = committed to git · 🔒 = gitignored (personal notes / large file).

> **New here? Suggested reading order:** `INSTALL_COLMAP_CUDA.md` (get it building) → `SFM_PIPELINE_COMPARISON.md` (what the preprocessing does) → `PIXEL_SHIFT_BUG.md` (the key correctness fix) → `FIP_PAPER_BENCH_RESULTS.md` (the headline results).

---

### 🛠️ Setup & environment
| Doc | | What it's about |
|---|---|---|
| [INSTALL_COLMAP_CUDA.md](INSTALL_COLMAP_CUDA.md) | ✅ | Build COLMAP from source with CUDA on a fresh machine — all deps, cmake config, GPU compute-capability table, and a fix for every build error we hit. |
| [euler_setup.md](euler_setup.md) | 🔒 | ETH Euler HPC **one-time setup**: connect via code-server, transfer code + data, install conda env + compile CUDA submodules, job-script contents (Steps 1–8). |
| [euler_setup_runs.md](euler_setup_runs.md) | 🔒 | ETH Euler **per-run workflow**: the push-code → connect → `sbatch` → pull-results cycle for training / segmentation / full-pipeline jobs (companion to `euler_setup.md`). |

### 🛰️ SfM preprocessing & Agisoft benchmark
| Doc | | What it's about |
|---|---|---|
| [SFM_PIPELINE_COMPARISON.md](SFM_PIPELINE_COMPARISON.md) | ✅ | Our COLMAP preprocessing vs the supervisor's Agisoft Metashape — the two functional gaps (metric scale, marker GCPs) and when to use which `sparse/`. |
| [AGISOFT_QUALITY_METRICS.md](AGISOFT_QUALITY_METRICS.md) | ✅ | How to read `3D Error` / `Distance Error` / `Reproj Error` in `marker_errors_summary.csv` — i.e. which Agisoft sessions to trust as references. |
| [COMPARE_TO_AGISOFT_RESULTS.md](COMPARE_TO_AGISOFT_RESULTS.md) | ✅ | 4-session benchmark of `compare_to_agisoft.py`: per-session translation/rotation error vs Agisoft, "good" thresholds, and the blurry `field_D/20250530` outlier. |

### 🎯 Pixel-shift fix (reconstruction correctness)
| Doc | | What it's about |
|---|---|---|
| [PIXEL_SHIFT_BUG.md](PIXEL_SHIFT_BUG.md) | ✅ | Vanilla 3DGS silently drops `cx, cy` → render/GT misalignment. The bug, per-plot offset table, why phone is unaffected, and three fix paths (`use_principal_point` is implemented). |
| [CAMERA_INTRINSICS_EXPLAINED.md](CAMERA_INTRINSICS_EXPLAINED.md) | 🔒 | Beginner primer on `fx/fy/cx/cy`, focal length, FoV, principal point, projection matrices and symmetric vs asymmetric frustums. Companion to `PIXEL_SHIFT_BUG.md`. |
| [FIP_PAPER_BENCH_RESULTS.md](FIP_PAPER_BENCH_RESULTS.md) | ✅ | 7-plot 30k PSNR/SSIM/LPIPS benchmark: the **+7.79 dB** jump from the pixel-shift fix, the eval-split regression diagnosis, and the head-to-head vs the original Wheat3DGS paper (we beat it). |

### ⚡ Render engine — gsplat port
| Doc | | What it's about |
|---|---|---|
| [GSPLAT_PORT.md](GSPLAT_PORT.md) | ✅ | The `diff-gaussian` → **gsplat** port (`gsplat-switch` branch): what changed in `render()`, the 3 correctness details, and what stays on flashsplat (segmentation). |
| [GSPLAT_VS_DIFFGS_RESULTS.md](GSPLAT_VS_DIFFGS_RESULTS.md) | ✅ | Head-to-head on 7 FIP plots: equal quality + equal segmentation but **~1.77× faster training** ⇒ a strict win. (FIP only — phone still TODO.) |

### 🧬 Training strategy — densification
| Doc | | What it's about |
|---|---|---|
| [DENSIFICATION_OPTIONS.md](DENSIFICATION_OPTIONS.md) | ✅ | Default vs **AbsGrad** vs **MCMC** densification: the "gradient-collision" blur on fine wheat detail (awns), the recommendation, the **implemented `absgrad` flag** (§8), and a **survey of other 3DGS variants** (Mip-Splatting, GaussianPro, Scaffold-GS, Deformable, LV-3DGS) with why they don't fit our segmentation/scene (§9). |

### 🔬 Input-data diagnostics (FIP vs phone)
| Doc | | What it's about |
|---|---|---|
| [SPARSENESS_ANALYSIS.md](SPARSENESS_ANALYSIS.md) | ✅ | SfM sparseness of all FIP plots + phone sessions — FIP and phone are sparse in *opposite* ways (narrow-angle/dense vs wide-angle/sparse); maps each to a densification recommendation. |
| [JPEG_QUALITY_ANALYSIS.md](JPEG_QUALITY_ANALYSIS.md) | ✅ | What JPEG does (chroma subsampling, 8×8 DCT, quantization) and its per-step impact — low for COLMAP/3DGS, high for SAM masks; **4:2:0 chroma is the real mask-edge cost**. |
| [MASK_SIZE_ANALYSIS.md](MASK_SIZE_ANALYSIS.md) | ✅ | SAM wheat-head mask sizes (FIP vs phone) and the `JPEG @Npx` edge-impact metric — plus why even a small per-mask error matters across thousands of densely-packed heads. |

### 📚 Reference & data
| Doc | | What it's about |
|---|---|---|
| [analysis_results/](analysis_results/) | ✅ | Raw machine-readable JSON outputs from the `src/analysis/` scripts that the diagnostic write-ups above cite. |
| `wheat3dgs_paper.pdf` | 🔒 | The original Wheat3DGS paper this thesis adapts to phone capture (large file, not committed). |
| [CHANGES.md](CHANGES.md) | 🔒 | Personal per-file change log (what changed and why) — dev notes, not project history. |

---

## Where other docs live

- **Supervisor reference material:** [`../reference/agisoft/`](../reference/agisoft/) — Agisoft scripts (steps 6/7/10) + coded-marker spec sheet. Read-only, not part of our pipeline.
- **Project-wide guidance for Claude Code:** [`../CLAUDE.md`](../CLAUDE.md) (workspace root).
- **Per-module READMEs** (`src/<module>/README.md`):
  - [`src/preprocessing/README.md`](../src/preprocessing/README.md) — phone preprocessing (uniform-size + COLMAP), full empirical test log
  - [`src/reconstruction/README.md`](../src/reconstruction/README.md) — 3DGS training, rendering, metrics
  - [`src/segmentation_3d/README.md`](../src/segmentation_3d/README.md) — FlashSplat 3D segmentation
  - [`src/mask_generation/yolo_sam_v1/README.md`](../src/mask_generation/yolo_sam_v1/README.md) — YOLO+SAM pipelining details
  - [`src/mask_generation/metrics/README.md`](../src/mask_generation/metrics/README.md) — YOLO evaluation metrics
