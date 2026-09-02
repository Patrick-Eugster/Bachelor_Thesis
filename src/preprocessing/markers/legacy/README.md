# markers/legacy — frozen working history

_Moved here 2026-09-02._

Superseded marker-detector versions and one-off prototypes/tests/debug scripts from the marker work,
kept in-repo as development history. The live marker pipeline lives one folder up in
[`../`](../) (v6 → v7_cct → v8_cct + the decode/scale/gcp steps).

⚠️ **These no longer run as-is.** They were written as flat siblings of the live scripts. Several
import the live scripts by bare name (e.g. `from cct_forced_decode import …`, `from cctdecode import …`),
and the others rely on their old sibling paths and Hydra config locations. Moving them down one folder
breaks that, so they are frozen records, not runnable tools.

- **detect_markers.py (v1), v2–v5** — early detector iterations, superseded by v6/v7/v8.
- **test_cct_phase0 / phase0b / phase1_forced, debug_cct_window** — CCT decode tests + debug.
- **marker_gcp_lomo** — the LOMO metric experiment (live pipeline uses marker_gcp_ba).
- **compare_v7_vs_agisoft, score_markers_vs_gt, overlay_agisoft_markers, make_fiducial_template,
  make_agisoft_marker_ctx** — one-off marker analyses / tooling.
