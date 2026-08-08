"""Write agisoft/logs/marker_points3d.json so the AGISOFT arm gets an ROI + marker-masked metric context.

metrics.py builds its ROI (plot region) + marker-masked passes by projecting 3D markers into each view,
reading <source>/logs/marker_points3d.json. That file only exists in our COLMAP frame, so the Agisoft arm
(source = agisoft/) has no markers → those passes are skipped and the two R1 arms aren't comparable on ROI.

This script (Route A) brings the markers into Agisoft's frame by applying the Umeyama similarity transform
that compare_to_agisoft.py already recovered (ours -> Agisoft:  X_agi = s * R @ X_colmap + t). The alignment
error is mm-scale, negligible for a coarse plot-region bbox. Prereqs (both in the session's logs/):
  - marker_points3d.json      (our COLMAP 3D markers)
  - compare_to_agisoft.json   (the s, R, t transform — run compare_to_agisoft.py first, on the SAME agisoft/)

Read-only except the one JSON it writes (agisoft/logs/marker_points3d.json).
"""
import json
import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocessing/make_agisoft_marker_ctx")
def main(cfg: DictConfig):
    """Transform our COLMAP 3D markers into Agisoft's frame and write agisoft/logs/marker_points3d.json."""
    print("--- make_agisoft_marker_ctx config ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------------")

    source_path = os.path.join(cfg.dataset.input_dir, str(cfg.field), str(cfg.plot))
    colmap_markers = os.path.join(source_path, "logs", "marker_points3d.json")
    compare_json = os.path.join(source_path, "logs", "compare_to_agisoft.json")
    out_dir = os.path.join(source_path, "agisoft", "logs")
    out_path = os.path.join(out_dir, "marker_points3d.json")

    for p, what in ((colmap_markers, "COLMAP markers"), (compare_json, "compare_to_agisoft")):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {what}: {p}  (run the marker triangulation / compare_to_agisoft first)")
    if os.path.exists(out_path) and not cfg.overwrite:
        raise SystemExit(f"{out_path} already exists — pass overwrite=true to replace it.")

    with open(colmap_markers) as f:
        markers = json.load(f)
    with open(compare_json) as f:
        align = json.load(f)["alignment"]

    # ours -> Agisoft:  X_agi = s * R @ X_colmap + t   (Umeyama, exactly as compare_to_agisoft applies it)
    s = float(align["scale"])
    R = np.asarray(align["rotation_matrix"], dtype=float)   # 3x3
    t = np.asarray(align["translation"], dtype=float)       # (3,)

    pts = markers.get("points3d", {})
    n = 0
    for mid, entry in pts.items():
        xyz = np.asarray(entry["xyz"], dtype=float)
        entry["xyz"] = (s * (R @ xyz) + t).tolist()         # transform in place, keep other fields
        n += 1

    # annotate provenance so it's clear this file is DERIVED, not an independent triangulation
    markers["frame"] = "agisoft"
    markers["derived_from"] = "logs/marker_points3d.json via compare_to_agisoft.json Umeyama (Route A)"
    markers["umeyama_scale"] = s

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(markers, f, indent=2)

    print("\n" + "=" * 56)
    print("      AGISOFT MARKER CONTEXT WRITTEN (Route A)")
    print("=" * 56)
    print(f"{'Session:':<22} {cfg.field}/{cfg.plot}")
    print(f"{'Markers transformed:':<22} {n}")
    print(f"{'Umeyama scale (our→m):':<22} {s:.6f}")
    print(f"{'Written to:':<22} {out_path}")
    print("=" * 56 + "\n")
    return {"n_markers": n, "out": out_path}


if __name__ == "__main__":
    main()
