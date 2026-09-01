"""Emit the per-plot LaTeX body for the conservative marker-exclusion table (Appendix C.2),
from docs/analysis_results/fip_seg_marker_excluded_eval.json. Shows all four available metrics
(precision, recall, F1, IoU) before vs after masking the Agisoft marker disks, for the
YOLOv5-at-1280 masks (the cited detector). Recall is unchanged since exclusion drops only false
positives. Tabular body only; caption hand-authored in main.tex. Read-only."""
import json
d = json.load(open("docs/analysis_results/fip_seg_marker_excluded_eval.json"))
rows = {r["plot"]: r for r in d["per_run"] if r["detector"] == "seg_yv5_1280"}
MET = ["precision", "recall", "f1", "iou"]
for p in sorted(rows):
    r = rows[p]
    cells = []
    for m in MET:
        cells += [f'{r["raw"][m]:.3f}', f'{r["marker_excluded"][m]:.3f}']
    print(f'{p[-3:]} & ' + ' & '.join(cells) + r' \\')
