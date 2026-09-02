# Vendored: CCTDecode

`CCTDecodeRelease.py`, `Support.py`, `DrawCCT.py` are vendored (copied) from the
public GitHub repository **[poxiao2/CCTDecode](https://github.com/poxiao2/CCTDecode)**
(commit on the `master` branch, files originally under `CCTDecode/`).

They detect and decode **circular coded targets (CCT)** — the same family as the
Agisoft 12-bit coded ground markers used in this project. We adopt the **decode core**
(ellipse-fit → affine-rectify → `CCT_or_not` validate → ring-sample → bit-decode) and
plan to replace only the **front-end** (global Otsu → adaptive / crop-seeded). See
`docs/preprocessing/markers/MARKER_DETECTION_CCT.md` for the full write-up.

## Local modifications
- **`CCTDecodeRelease.py`**: made the `from progress.bar import ShadyBar` import lazy
  (moved from module top into `scanne_data_dir`), so importing the decode core does not
  require the `progress` package. No algorithm change.

## Licensing note
The upstream repository did **not** include an explicit LICENSE file at the time of
vendoring. These files are reused here for **academic research** (a bachelor thesis).
Verify the upstream licensing before any redistribution or non-academic use.
