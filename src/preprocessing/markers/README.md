# Coded Markers — `src/preprocessing/markers/`

The phone captures include coded ground markers, the same 12-bit CCT targets Agisoft uses, placed
in the field and visible across many images. These steps detect the coded markers and decode their
IDs, triangulate them to one 3D point each, and can derive a metric scale from a survey or from
tape measurements. They are **off by default** (`run_markers=false`) and specific to our capture
setup, so this is documented mainly for completeness and reproducibility rather than as a
general-purpose tool. Like the SfM steps, they run through the `run_preprocessing.py` orchestrator
and share `field=` and `plot=`.

## When you would run it

Two things use the markers, and the default sub-toggles reflect that:

- **Detection and triangulation** are on whenever `run_markers=true`. They produce
  `marker_points3d.json`, the 3D marker positions behind the marker-based region of interest used
  by phone mask generation and 3D segmentation.
- **Scale, metric model and GCP-BA** are off by default. They give the reconstruction a real
  metric size, which only phenotyping needs, and they require our field's survey file or tape
  measurements, which are not part of the public repo.

## How to run

```bash
# detect + triangulate only (the marker positions for the ROI)
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715 run_markers=true

# also derive a metric scale from the surveyed marker coordinates
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715 run_markers=true \
  run_marker_scale=true run_marker_metric=true run_marker_gcp=true marker_scale_source=survey

# metric size from tape distances only, without a survey (GCP-BA is skipped automatically)
python src/preprocessing/run_preprocessing.py field=field_A plot=20250715 run_markers=true \
  run_marker_scale=true run_marker_metric=true marker_scale_source=tape
```

The steps are fail-soft, so a missing survey or tape file warns and continues instead of aborting
the run.

## The steps

| Step | Script | What it does | Output |
|------|--------|--------------|--------|
| Detect | `detect_markers_v8_cct.py` | Decode the coded targets in each image (CCT decode) | `logs/marker_detections_v8_manifest.json` |
| Triangulate | `triangulate_markers.py` | Lift the detections to one 3D point per marker using the COLMAP poses | `logs/marker_points3d.json` |
| Scale | `marker_scale.py` | Recover the metric scale from the surveyed coordinates or tape distances | `logs/marker_scale.json` |
| Metric model (Flavour 1) | `apply_metric_transform.py` | Rewrite the model into a metric frame | `sparse_metric/` |
| GCP-BA (Flavour 2) | `marker_gcp_ba.py` | Bundle adjustment pinning the markers to the survey | `sparse_metric_gcp/` |

Detection and triangulation work in the undistorted `images/` frame, the same frame the camera
poses, 3DGS and 3D segmentation use.

## Two metric flavours

The metric steps can write two independent metric models, and you keep whichever fits:

- **Flavour 1 — `apply_metric_transform.py` → `sparse_metric/`.** Applies one similarity transform
  (scale, rotation, translation) to the whole model. The geometry is unchanged and only the
  coordinates become metric. With `marker_scale_source=survey` the model is also georeferenced,
  so it sits at its real-world position and orientation. With `tape` only the size is metric and
  the position and orientation stay arbitrary.
- **Flavour 2 — `marker_gcp_ba.py` → `sparse_metric_gcp/`.** A GCP-constrained bundle adjustment
  that pins the markers to their surveyed positions and re-optimizes the camera poses to honour
  them. It is survey-anchored, so it needs `pycolmap` and is skipped automatically in tape mode.

The **tape gate** (off by default, `tape_gate=true`) picks between the two automatically from how
well the tape distances and the survey agree, since anchoring to the survey helps when the survey
is good and hurts when it is off.

## Survey versus tape

`marker_scale_source` chooses where the metric scale comes from:

- **survey** (default) — surveyed marker coordinates (total-station or RTK). The result is metric
  and georeferenced, so the model has real size and sits at its real-world position and orientation.
- **tape** — pairwise tape distances only. The result has correct metric size, but its position
  and orientation in the world stay arbitrary, which is all phenotyping needs.

## Failsafe

The metric steps only run when triangulation solved at least `min_markers` markers (default 4)
that pass a quality gate on triangulation angle, inlier views and reprojection error. With fewer
good markers the run keeps the reconstruction in its original relative scale rather than anchoring
metric size on one or two weak points.

## Markers in SfM (experimental)

`markers_in_sfm=true` builds the markers into the reconstruction from the start. A detection pass
runs on `input_uniform` before COLMAP, and the decoded markers are injected as extra image
tie-points between matching and the mapper, so a fresh SfM uses them as guaranteed correspondences
while it builds the model. Unlike Flavour 2, which pins the markers to their surveyed positions
after the reconstruction is built, this route needs no survey, since the markers act only as extra
correspondences between images. It is experimental and off by default.

## Scripts and layout

- **Steps that run:** `detect_markers_v8_cct.py`, `triangulate_markers.py`, `marker_scale.py`,
  `apply_metric_transform.py`, `marker_gcp_ba.py`.
- **Helper modules the v8 detector is built on:** `detect_markers_v6.py`,
  `detect_markers_v7_cct.py`, `cct_forced_decode.py`, `marker_codes.py`, and `cctdecode/`. The v8
  detector imports these at runtime, so they are live dependencies, not old versions.
- **`inject_markers_to_db.py`** — the helper used by the experimental markers-in-SfM route.
- **`legacy/`** — superseded detectors (v1 to v5) and one-off analysis tools, kept as frozen
  development history. They no longer run as-is. See [`legacy/README.md`](legacy/README.md).

## Evaluating the markers and the SfM geometry

None of the marker evaluations are part of a run. They are standalone scripts in
[`src/analysis/`](../../analysis/):

- **`eval_marker_detection.py`** — detector accuracy in 2D per session, meaning recall, precision
  and how far the detected center sits from the hand-verified marker pin.
- **`marker_geometry_gt.py`** — checks the triangulated marker geometry against the surveyed
  coordinates and the tape measurements, using all pairwise marker distances.
- **`eval_marker_geometry_gt.py`** — the same geometry error without the detector in the way,
  by triangulating the verified pins through our poses.
- **`rescore_models_geometry.py`** — scores different SfM front-ends by their marker geometry
  error in centimetres. This is the "which front-end is best" evaluation.
- **`compare_sfm_models_markers.py`** — compares SfM models by marker reprojection error against
  the Agisoft reference, with a shared triangulation so the comparison is fair.
- **`marker_cross_session_repeatability.py`** — how repeatable the triangulated marker positions
  are across sessions.

They read the survey coordinates and tape measurements from our private reference data, which is not
part of this repo, so they only run where that data is present. Read the notes at the top of
[`src/analysis/README.md`](../../analysis/README.md) before running one.

> The full marker write-ups (the CCT decode, the integration plan, the survey-versus-tape study)
> live under `docs/`, a private working-notes submodule that is not part of a public clone.
