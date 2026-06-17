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

### 🌾 Detection & masks (YOLO + SAM)
| Doc | | What it's about |
|---|---|---|
| [MASK_GENERATION_OPTIONS.md](MASK_GENERATION_OPTIONS.md) | ✅ | Survey of detector + SAM alternatives for the wheat-head mask stage: why no turnkey detector beats our GWC YOLOv5 weights without training, **SAHI** as the top training-free lever for the dense/small phone heads, SAM2/SAM3/HQ-SAM mask upgrades, and the phone-vs-FIP diagnosis (density + JPG, *not* viewpoint). |
| [SAHI_EXPLAINED.md](SAHI_EXPLAINED.md) | ✅ | From-scratch SAHI walkthrough: the downscale problem, the tile-count formula (worked example), how to **avoid scaling** (set `imgsz = slice`), the edge-split → overlap fix (+ our 161 px head caveat), the detected-twice → NMM/IOS merge, the full-image safety net, and quick-start settings. |
| [SAHI_EVAL_RESULTS.md](SAHI_EVAL_RESULTS.md) | ✅ | SAHI-vs-YOLO eval results: the **confidence-floor bug** that made SAHI first look worse (NMM unions junk boxes at floor 0.01; YOLO immune), the **single-threshold fix** (SAHI now beats YOLO on recall, +0.04 F1), and the **IOS/IOU/CONF merge study** (IOS wins; the nested-head case is not box-fixable → mask-based dedup, parked for phone). |
| [SAHI_MASK_DEDUP.md](SAHI_MASK_DEDUP.md) | ✅ | The **experimental mask-based dedup** (standalone `sahi_mask_dedup.py`, production untouched): how it works (no box-merge → SAM clean per-head masks via point+negative prompts → dedup on *mask* overlap), the knobs, honest results (doesn't beat IOS — the nested heads are a *detection* miss, not a merge miss), and the better **surgical/hybrid** TODO (SAM only on ambiguous overlapping clusters). |
| [SAHI_SURGICAL_DEDUP.md](SAHI_SURGICAL_DEDUP.md) | ✅ | The **surgical/hybrid dedup** (standalone `sahi_surgical_dedup.py`, production + v1 untouched): the "better future approach" from the v1 doc, now built — box logic handles the clean majority (tier-1 keep / tier-2 IoU-NMS, no SAM), **SAM only tie-breaks the ambiguous contained pairs** (tier-3). The core difference-from-v1 table + the 3-tier flow diagram. |

### 🔬 Input-data diagnostics (FIP vs phone)
| Doc | | What it's about |
|---|---|---|
| [SPARSENESS_ANALYSIS.md](SPARSENESS_ANALYSIS.md) | ✅ | SfM sparseness of all FIP plots + phone sessions — FIP and phone are sparse in *opposite* ways (narrow-angle/dense vs wide-angle/sparse); maps each to a densification recommendation. |
| [JPEG_QUALITY_ANALYSIS.md](JPEG_QUALITY_ANALYSIS.md) | ✅ | What JPEG does (chroma subsampling, 8×8 DCT, quantization) and its per-step impact — low for COLMAP/3DGS, high for SAM masks; **4:2:0 chroma is the real mask-edge cost**. |
| [MASK_SIZE_ANALYSIS.md](MASK_SIZE_ANALYSIS.md) | ✅ | SAM wheat-head mask sizes (FIP vs phone) and the `JPEG @Npx` edge-impact metric — plus why even a small per-mask error matters across thousands of densely-packed heads. |

### 📚 Reference & data
| Doc | | What it's about |
|---|---|---|
| [analysis_results/](analysis_results/) | 🔒 | Raw machine-readable JSON outputs from the `src/analysis/` scripts that the diagnostic write-ups above cite. The folder is kept in git (via its own `.gitignore`) but its **contents are gitignored** — they're regenerable, so re-run the `src/analysis/` scripts to recreate them. |
| `wheat3dgs_paper.pdf` | 🔒 | The original Wheat3DGS paper this thesis adapts to phone capture (large file, not committed). |
| [CHANGES.md](CHANGES.md) | 🔒 | Personal per-file change log (what changed and why) — dev notes, not project history. |

---

## Where other docs live

- **Archived plans** (`docs/archive/`, gitignored — local-only, kept for the record): `SAHI_IMPLEMENTATION_PLAN.md` and `SAHI_YOLO_EVAL_PLAN.md` were the design plans for the SAHI detector and the SAHI-vs-YOLO eval tools. Both are now **built** — see `SAHI_EXPLAINED.md` and `src/mask_generation/evaluation/README.md` for the as-built docs.
- **Supervisor reference material:** [`../reference/agisoft/`](../reference/agisoft/) — Agisoft scripts (steps 6/7/10) + coded-marker spec sheet. Read-only, not part of our pipeline.
- **Project-wide guidance for Claude Code:** [`../CLAUDE.md`](../CLAUDE.md) (workspace root).
- **Per-module READMEs** (`src/<module>/README.md`):
  - [`src/preprocessing/README.md`](../src/preprocessing/README.md) — phone preprocessing (uniform-size + COLMAP), full empirical test log
  - [`src/reconstruction/README.md`](../src/reconstruction/README.md) — 3DGS training, rendering, metrics
  - [`src/segmentation_3d/README.md`](../src/segmentation_3d/README.md) — FlashSplat 3D segmentation
  - [`src/mask_generation/yolo_sam_v1/README.md`](../src/mask_generation/yolo_sam_v1/README.md) — YOLO+SAM pipelining details
  - [`src/mask_generation/evaluation/README.md`](../src/mask_generation/evaluation/README.md) — YOLO evaluation metrics
