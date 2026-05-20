# reference/agisoft/ — supervisor's Agisoft material

Read-only reference. **These are not part of our pipeline** — we do not import, run, or
maintain anything here. They live in the repo so we can understand how the supervisor's
Agisoft processing works and how the benchmark data we compare against was produced.

For the prose docs that interpret this material — what the marker-error CSV means, how
the supervisor pipeline maps to ours, etc. — see [`../../docs/`](../../docs/).

## Files

| File | What it is |
|---|---|
| `6-agisoft_preprocessing_demoanlage_2025.py` | Supervisor's **step 6** — runs Agisoft Metashape on the raw OpenCamera phone captures (`field_*/<date>/video/*.mp4` → extracted frames → Agisoft project → exported `sparse/`). Produces the `agisoft/` folder we use as ground truth in our compare step. |
| `7-agisoft_compute_marker_errors.py` | Supervisor's **step 7** — opens the saved Agisoft project, reads the coded-marker detections and their fitted 3D positions, and writes the per-session `marker_errors_summary.csv` (3D Error / Distance Error / Reproj Error). Determines which Agisoft sessions are clean enough to use as a benchmark reference. See [`../../docs/AGISOFT_QUALITY_METRICS.md`](../../docs/AGISOFT_QUALITY_METRICS.md) for what those columns mean. |
| `10_agisoft_calibration_quality_analysis.ipynb` | Supervisor's **step 10** — Jupyter notebook that reads `marker_errors_summary.csv` and plots per-session calibration quality (which captures are precise enough, which to discard). |
| `Coded_12bit_15cm-square_13cm-outer-circle_.pdf` | Spec sheet for the **coded ground markers** physically placed in the field for Agisoft to detect. 12-bit codes printed on a 15 cm square with a 13 cm outer circle. This is what Agisoft uses to anchor the reconstruction to a metric, known scale — see [`../../docs/SFM_PIPELINE_COMPARISON.md`](../../docs/SFM_PIPELINE_COMPARISON.md) for why our COLMAP pipeline currently lacks this and what closing the gap would take. |

## How this relates to our code

- Our pipeline ([`../../src/preprocessing/`](../../src/preprocessing/)) reproduces Agisoft's role
  using open-source COLMAP — output goes to `{plot}/sparse/0/` (ours) vs `{plot}/agisoft/sparse/0/`
  (supervisor's reference, produced by script 6 above).
- [`../../src/preprocessing/compare_to_agisoft.py`](../../src/preprocessing/compare_to_agisoft.py)
  is the script that benchmarks the two. The 4-session results are in
  [`../../docs/COMPARE_TO_AGISOFT_RESULTS.md`](../../docs/COMPARE_TO_AGISOFT_RESULTS.md).
- Numbering (6, 7, 10) is the supervisor's own pipeline step numbers — kept as-is so
  the filenames match the prose in `docs/SFM_PIPELINE_COMPARISON.md`.
