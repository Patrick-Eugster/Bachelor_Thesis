"""Route 2 — inject decoded markers into a COLMAP database as guaranteed TIE-POINTS.

After `feature_extractor` + the matcher have populated `distorted/database.db` with SIFT keypoints and
verified matches, this adds the 2D marker observations as EXTRA keypoints and links the SAME marker id
across images as EXTRA verified matches (`two_view_geometries`, what the mapper consumes). The mapper
then triangulates the markers AS PART of the SfM — a full second reconstruction with markers baked in.

TIE-POINTS ONLY: we add image-to-image correspondences ("this pixel in image A is the same physical
marker as this pixel in image B"), NOT world coordinates — so it is completely survey-free / GPS-free.
A marker seen in N images contributes N keypoints + every cross-image pair as a guaranteed match, so it
forms one track and comes out as a single 3D point. Metric scale still comes post-hoc from tape
(marker_scale.py scale_source=tape). See docs/preprocessing/markers/MARKER_INTEGRATION_PLAN.md (Route 2).

The marker pixels MUST be detected on the SAME images COLMAP extracted features from (the distorted
`input_uniform/` for phone) so the coordinates line up with the SIFT keypoints. Run detection with
image_subdir=input_uniform before injecting.

We talk to the COLMAP SQLite schema directly (pycolmap 4.0.4's Database is an abstract base that
segfaults); the schema is stable: keypoints(image_id, rows, cols, data float32), matches +
two_view_geometries(pair_id, rows, cols, data uint32), pair_id = id1*2147483647 + id2 (id1<id2).

Usage:
    python src/preprocessing/inject_markers_to_db.py field=field_A plot=20250609
"""

import json
import os
import sqlite3
from collections import defaultdict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

MAX_IMAGE_ID = 2147483647
CONFIG_CALIBRATED = 2  # two_view_geometries.config — a normal verified two-view geometry


def pair_id(id1, id2):
    """COLMAP image-pair id, with the smaller image id first (the schema's convention)."""
    if id1 > id2:
        id1, id2 = id2, id1
    return id1 * MAX_IMAGE_ID + id2


def _to_array(blob, dtype, cols):
    """COLMAP blob -> (N, cols) array (copy, so it's writable); None/empty -> (0, cols)."""
    if not blob:
        return np.zeros((0, cols), dtype=dtype)
    return np.frombuffer(blob, dtype=dtype).reshape(-1, cols).copy()


def load_marker_obs(detections_json):
    """detect_markers_v8 per_image JSON -> {image_name: {marker_id: (x, y)}}."""
    d = json.load(open(detections_json))
    out = {}
    for name, dets in d["per_image"].items():
        for det in dets:
            x, y = det["center"]
            out.setdefault(name, {})[int(det["id"])] = (float(x), float(y))
    return out


def _append_keypoints(cur, iid, markers):
    """Append marker pixels as keypoints for one image; return {marker_id: kp_index} (or {} if none).

    Keeps the descriptors table row-aligned (pads zeros) so rows match keypoints — the matcher already
    ran, so the zero descriptors are never used, but equal counts avoid any consistency check."""
    row = cur.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (iid,)).fetchone()
    if row is None:
        return {}
    rows, cols, data = row
    kps = _to_array(data, np.float32, cols)
    index, new = {}, []
    for mid, (x, y) in sorted(markers.items()):
        # match the existing column count: 6=[x,y,affine 2x2], 4=[x,y,scale,orient], 2=[x,y]
        if cols == 6:
            r = [x, y, 1, 0, 0, 1]
        elif cols == 4:
            r = [x, y, 1, 0]
        else:
            r = [x, y][:cols]
        index[mid] = rows + len(new)
        new.append(r)
    if not new:
        return {}
    kps2 = np.vstack([kps, np.asarray(new, dtype=np.float32)])
    cur.execute("UPDATE keypoints SET rows=?, data=? WHERE image_id=?",
                (kps2.shape[0], kps2.tobytes(), iid))
    drow = cur.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (iid,)).fetchone()
    if drow is not None:
        drows, dcols, ddata = drow
        desc = _to_array(ddata, np.uint8, dcols)
        desc2 = np.vstack([desc, np.zeros((len(new), dcols), dtype=np.uint8)])
        cur.execute("UPDATE descriptors SET rows=?, data=? WHERE image_id=?",
                    (desc2.shape[0], desc2.tobytes(), iid))
    return index


def _append_matches(cur, table, pid, add):
    """Append (kp_idx1, kp_idx2) rows to matches/two_view_geometries for one pair (insert if new)."""
    row = cur.execute(f"SELECT rows, data FROM {table} WHERE pair_id=?", (pid,)).fetchone()
    if row is None:
        if table == "two_view_geometries":
            cur.execute("INSERT INTO two_view_geometries(pair_id, rows, cols, data, config) "
                        "VALUES (?,?,?,?,?)", (pid, add.shape[0], 2, add.tobytes(), CONFIG_CALIBRATED))
        else:
            cur.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES (?,?,?,?)",
                        (pid, add.shape[0], 2, add.tobytes()))
    else:
        both = np.vstack([_to_array(row[1], np.uint32, 2), add])
        cur.execute(f"UPDATE {table} SET rows=?, data=? WHERE pair_id=?",
                    (both.shape[0], both.tobytes(), pid))


def inject(db_path, detections_json):
    """Inject marker tie-points into a COLMAP database. Returns a small summary dict."""
    obs = load_marker_obs(detections_json)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    name_to_id = {n: i for (i, n) in cur.execute("SELECT image_id, name FROM images")}

    # 1. add each image's marker pixels as keypoints; remember (image_id, marker_id) -> kp index
    kp_index, skipped = {}, 0
    for name, markers in obs.items():
        if name not in name_to_id:   # detected on an image colmap didn't register/extract — skip
            skipped += 1
            continue
        iid = name_to_id[name]
        for mid, idx in _append_keypoints(cur, iid, markers).items():
            kp_index[(iid, mid)] = idx

    # 2. for every image-pair that shares a marker id, add a guaranteed match
    marker_to_imgs = defaultdict(list)
    for (iid, mid) in kp_index:
        marker_to_imgs[mid].append(iid)
    n_pairs = n_edges = 0
    for mid, imgs in marker_to_imgs.items():
        imgs = sorted(set(imgs))
        for a in range(len(imgs)):
            for b in range(a + 1, len(imgs)):
                i1, i2 = imgs[a], imgs[b]          # i1 < i2, so kp index for i1 goes first
                add = np.array([[kp_index[(i1, mid)], kp_index[(i2, mid)]]], dtype=np.uint32)
                pid = pair_id(i1, i2)
                _append_matches(cur, "matches", pid, add)
                _append_matches(cur, "two_view_geometries", pid, add)
                n_pairs += 1
                n_edges += 1
    con.commit()
    con.close()
    return {
        "markers_injected": len(kp_index),
        "markers_seen": len(marker_to_imgs),
        "match_edges": n_edges,
        "pairs_touched": n_pairs,
        "images_skipped_not_in_db": skipped,
    }


@hydra.main(version_base=None, config_path="../../../configs/preprocessing", config_name="inject_markers")
def main(cfg: DictConfig):
    """Standalone: inject marker tie-points into a session's distorted/database.db."""
    print(OmegaConf.to_yaml(cfg))
    db_path = os.path.join(cfg.source_path, cfg.database_path)
    det_json = os.path.join(cfg.source_path, cfg.detections_json)
    for p in (db_path, det_json):
        if not os.path.isfile(p):
            raise SystemExit(f"missing input: {p}")
    summary = inject(db_path, det_json)
    print("\n" + "=" * 60)
    print(f"  MARKER INJECTION  {cfg.field}/{cfg.plot}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<26} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
