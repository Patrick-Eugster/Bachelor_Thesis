"""Point-GT tool — local web server (stdlib only, no FastAPI/Flask so we add zero dependencies).

Serves a canvas UI to correct a SAM draft into pixel ground truth. Flow: pick a GT image -> the server
SAM-seeds every YOLO/SAHI box into a draft mask -> you select bad heads and fix them with positive/negative
point clicks (per-head crop) -> save writes <stem>_gt_mask.png etc into that session's manual_label/.

Run (in the container):  python -m mask_generation.gt_tool.server
Then open http://localhost:8000 in a browser.

Single user / single active image by design. All SAM calls go through one GPU under a lock.
"""

import os
import io
import sys
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2
from PIL import Image

# repo paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PHONE = os.path.join(REPO, "input_plots", "phone")
STATIC = os.path.join(HERE, "static")
SELECTION = os.path.join(PHONE, "gt_selection.json")

# config (env-overridable)
WEIGHT = os.environ.get("GT_SAM_WEIGHT",
                        os.path.join(HERE, "..", "weights", "sam2.1_l.pt"))   # src/mask_generation/weights/
DECODE_BATCH = int(os.environ.get("GT_DECODE_BATCH", "8"))
PORT = int(os.environ.get("GT_PORT", "8000"))

_engine = None            # lazy SamEngine
_lock = threading.Lock()  # serialize GPU access


def get_engine():
    """Load the SAM engine once, on first use (so `--help`/list don't pay the model load)."""
    global _engine
    if _engine is None:
        from mask_generation.gt_tool.sam_prompt import SamEngine
        print(f"[gt_tool] loading SAM weight {WEIGHT} (decode_batch={DECODE_BATCH}) ...", flush=True)
        _engine = SamEngine(WEIGHT, decode_batch=DECODE_BATCH)
        print("[gt_tool] SAM ready.", flush=True)
    return _engine


def color_for(i):
    """Distinct BGR color per instance id via golden-ratio hue spacing."""
    h = int((i * 0.61803398875 % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def to_sparse(full_mask):
    """Store a mask as (bbox, small submask) so overlay render + hit-test only touch its box region.
    Returns None for an empty mask."""
    ys, xs = np.where(full_mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return {"bbox": [x0, y0, x1, y1], "sub": np.ascontiguousarray(full_mask[y0:y1, x0:x1])}


class Session:
    """Holds the one active image and its named mask-SETS. `instances`/`next_id` are properties that proxy
    to the ACTIVE set, so all the rest of the code works on the active set without changes. Sets let you
    keep e.g. a YOLO-SAM set and a manual set side by side and switch between them."""

    def __init__(self):
        self.field = self.date = self.stem = None
        self.img = None                 # RGB uint8 HxW3
        self.sets = {}                  # {name: {"instances": [...], "next_id": int}}
        self.active = None              # active set name
        self.pending_boxes = []         # seed boxes waiting for the (separate) seed pass
        self.backup = []                # snapshot kept by Clear so it can be undone
        self.reset_sets()

    def reset_sets(self):
        """Start fresh with a single empty set (used when a new image loads)."""
        self.sets = {"set 1": {"instances": [], "next_id": 1}}
        self.active = "set 1"
        self.backup = []

    @property
    def instances(self):
        return self.sets[self.active]["instances"]

    @instances.setter
    def instances(self, v):
        self.sets[self.active]["instances"] = v

    @property
    def next_id(self):
        return self.sets[self.active]["next_id"]

    @next_id.setter
    def next_id(self, v):
        self.sets[self.active]["next_id"] = v

    def add_set(self):
        """Create a new empty set, make it active, return its name."""
        n = 1
        while f"set {n}" in self.sets:
            n += 1
        name = f"set {n}"
        self.sets[name] = {"instances": [], "next_id": 1}
        self.active = name
        self.backup = []
        return name

    def sets_info(self):
        """List of {name, n} + active — for the client's set dropdown."""
        return {"sets": [{"name": k, "n": len(v["instances"])} for k, v in self.sets.items()],
                "active": self.active}

    def add(self, sparse, seed_box=None, points=None):
        """Register a new instance in the active set, return its id."""
        iid = self.next_id
        self.next_id = iid + 1
        self.instances.append({"id": iid, "seed_box": seed_box, "mask": sparse,
                               "points": points or [], "hidden": False, "locked": False})
        return iid

    def find(self, iid):
        """Instance dict by id in the active set, or None."""
        return next((it for it in self.instances if it["id"] == iid), None)


SESS = Session()


def _load_seed_boxes(field, date, stem, W, H):
    """Read the YOLO-normalized seed boxes for this image -> list of [x0,y0,x1,y1] pixels."""
    p = os.path.join(PHONE, field, date, "gt_labeling", f"{stem}.txt")
    boxes = []
    if os.path.exists(p):
        for ln in open(p):
            parts = ln.split()
            if len(parts) != 5:
                continue
            _c, cx, cy, w, h = map(float, parts)
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
    return boxes


def _cache_dir():
    """Where the SAM-seeded draft is cached (so re-opening an image doesn't re-decode all masks)."""
    return os.path.join(PHONE, SESS.field, SESS.date, "gt_cache")


def _save_dir():
    """Where the finished GT (and its resumable state) is written."""
    return os.path.join(PHONE, SESS.field, SESS.date, "manual_label")


def _dump_state(out_dir):
    """Persist the current instances as a uint16 instance map + JSON (id/seed_box/points) so they can be
    reloaded instantly. Decodes happen once; this is the cache that makes re-opening fast."""
    os.makedirs(out_dir, exist_ok=True)
    H, W = SESS.img.shape[:2]
    inst_map = np.zeros((H, W), np.uint16)
    meta_inst = []
    for it in SESS.instances:
        sp = it["mask"]
        if sp is None:
            continue
        x0, y0, x1, y1 = sp["bbox"]
        inst_map[y0:y1, x0:x1][sp["sub"]] = it["id"]
        meta_inst.append({"id": it["id"], "bbox": [x0, y0, x1, y1], "seed_box": it["seed_box"],
                          "points": it["points"], "locked": it["locked"]})
    cv2.imwrite(os.path.join(out_dir, f"{SESS.stem}_instances.png"), inst_map)
    json.dump({"stem": SESS.stem, "next_id": SESS.next_id, "backend": os.path.basename(WEIGHT),
               "instances": meta_inst},
              open(os.path.join(out_dir, f"{SESS.stem}_seed.json"), "w"), indent=1)


def _load_state(in_dir):
    """Rebuild instances from a previously dumped instance map + JSON. Returns True if a cache existed.
    Reconstructs each mask as (instance_map == id), so no SAM decode is needed."""
    png = os.path.join(in_dir, f"{SESS.stem}_instances.png")
    js = os.path.join(in_dir, f"{SESS.stem}_seed.json")
    if not (os.path.exists(png) and os.path.exists(js)):
        return False
    inst_map = cv2.imread(png, cv2.IMREAD_UNCHANGED)     # uint16 label map
    meta = json.load(open(js))
    SESS.instances = []
    for m in meta["instances"]:
        if "bbox" in m:                                  # fast path: only scan the instance's own box
            x0, y0, x1, y1 = m["bbox"]
            sub = np.ascontiguousarray(inst_map[y0:y1, x0:x1] == m["id"])
            sp = {"bbox": [x0, y0, x1, y1], "sub": sub} if sub.any() else None
        else:
            sp = to_sparse(inst_map == m["id"])          # fallback (old caches without bbox)
        if sp is not None:
            SESS.instances.append({"id": m["id"], "seed_box": m["seed_box"], "mask": sp,
                                   "points": m.get("points", []),
                                   "hidden": False, "locked": bool(m.get("locked", False))})
    ids = [m["id"] for m in meta["instances"]]
    SESS.next_id = meta.get("next_id", (max(ids) + 1) if ids else 1)
    SESS.pending_boxes = []
    return True


def _dump_instances(instances, next_id, png_path, json_path):
    """Write one set's instances to (png, json). Same format as _dump_state but for an arbitrary set."""
    H, W = SESS.img.shape[:2]
    inst_map = np.zeros((H, W), np.uint16)
    meta_inst = []
    for it in instances:
        sp = it["mask"]
        if sp is None:
            continue
        x0, y0, x1, y1 = sp["bbox"]
        inst_map[y0:y1, x0:x1][sp["sub"]] = it["id"]
        meta_inst.append({"id": it["id"], "bbox": [x0, y0, x1, y1], "seed_box": it["seed_box"],
                          "points": it["points"], "locked": it["locked"]})
    cv2.imwrite(png_path, inst_map)
    json.dump({"stem": SESS.stem, "next_id": next_id, "instances": meta_inst},
              open(json_path, "w"), indent=1)


def _load_instances(png_path, json_path):
    """Read one set back from (png, json). Returns (instances_list, next_id) or None."""
    if not (os.path.exists(png_path) and os.path.exists(json_path)):
        return None
    inst_map = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    meta = json.load(open(json_path))
    insts = []
    for m in meta["instances"]:
        x0, y0, x1, y1 = m["bbox"]
        sub = np.ascontiguousarray(inst_map[y0:y1, x0:x1] == m["id"])
        if sub.any():
            insts.append({"id": m["id"], "seed_box": m["seed_box"],
                          "mask": {"bbox": [x0, y0, x1, y1], "sub": sub},
                          "points": m.get("points", []), "hidden": False,
                          "locked": bool(m.get("locked", False))})
    ids = [m["id"] for m in meta["instances"]]
    return insts, meta.get("next_id", (max(ids) + 1) if ids else 1)


def _save_all_sets():
    """Persist EVERY mask-set to manual_label/<stem>_sets/ + a manifest, so all sets (incl. backups from
    Clear) survive a reload. The active set's union is what eval reads as the GT."""
    sets_dir = os.path.join(_save_dir(), f"{SESS.stem}_sets")
    if os.path.isdir(sets_dir):
        for f in os.listdir(sets_dir):
            os.remove(os.path.join(sets_dir, f))          # drop stale set files from a previous save
    os.makedirs(sets_dir, exist_ok=True)
    manifest = {"active": SESS.active, "sets": []}
    for i, (name, s) in enumerate(SESS.sets.items()):
        _dump_instances(s["instances"], s["next_id"],
                        os.path.join(sets_dir, f"set{i}_instances.png"),
                        os.path.join(sets_dir, f"set{i}_seed.json"))
        manifest["sets"].append({"name": name, "file": f"set{i}"})
    json.dump(manifest, open(os.path.join(sets_dir, "manifest.json"), "w"), indent=1)


def _load_all_sets():
    """Restore every mask-set from manual_label/<stem>_sets/. Returns True if a saved set-state existed."""
    sets_dir = os.path.join(_save_dir(), f"{SESS.stem}_sets")
    mp = os.path.join(sets_dir, "manifest.json")
    if not os.path.exists(mp):
        return False
    manifest = json.load(open(mp))
    sets = {}
    for entry in manifest["sets"]:
        r = _load_instances(os.path.join(sets_dir, f"{entry['file']}_instances.png"),
                            os.path.join(sets_dir, f"{entry['file']}_seed.json"))
        if r is not None:
            sets[entry["name"]] = {"instances": r[0], "next_id": r[1]}
    if not sets:
        return False
    SESS.sets = sets
    SESS.active = manifest["active"] if manifest.get("active") in sets else next(iter(sets))
    return True


ROI_BUFFER_FRAC = 0.05   # matches make_gt_labeling_images.py's grey-out buffer (roi.buffer_frac default)


def compute_roi_poly():
    """The ROI boundary for the current image (in images/ pixel space), or None if the plot has no full
    marker ring (roi_mask disables it). Returns the BUFFERED region's outline — the marker hull grown by
    buffer_frac, i.e. exactly the boundary the pipeline greyed out when the seed boxes were made (so the
    border lines up with reality, not the un-buffered hull)."""
    try:
        from mask_generation import roi_mask
        plot_dir = os.path.join(PHONE, SESS.field, SESS.date)
        poly = roi_mask._build_plot_polys(plot_dir, 3).get(f"{SESS.stem}.jpg")
        if poly is None:
            return None
        H, W = SESS.img.shape[:2]
        buffer_px = int(round(ROI_BUFFER_FRAC * min(W, H)))
        keep = roi_mask._roi_keep_region(poly, W, H, buffer_px).astype(np.uint8)   # buffered kept region
        cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        big = max(cnts, key=cv2.contourArea)
        eps = 0.002 * cv2.arcLength(big, True)                # simplify (keeps the rounded corners roughly)
        return cv2.approxPolyDP(big, eps, True).reshape(-1, 2).tolist()
    except Exception as e:
        print(f"[gt_tool] ROI unavailable: {e}")
        return None


def load_image(field, date, stem, auto_seed=True):
    """Load the image and restore SAVED work (manual_label) — that ALWAYS comes back and is never touched.
    If there's no saved work: with auto_seed on, restore the cached seed draft (fast) if present; with
    auto_seed off, start BLANK (0 masks) so you can label manually. The YOLO+SAM seeds can be pulled in
    any time with add_seeds() — which only APPENDS, never deletes."""
    path = os.path.join(PHONE, field, date, "images", f"{stem}.jpg")
    img = np.array(Image.open(path).convert("RGB"))         # PIL = silent on Samsung JPEGs
    H, W = img.shape[:2]

    SESS.field, SESS.date, SESS.stem = field, date, stem
    SESS.img = img
    SESS.reset_sets()
    SESS.pending_boxes = []

    out = {"w": W, "h": H, "field": field, "date": date, "stem": stem, "roi": compute_roi_poly()}
    if _load_all_sets():                                    # SAVED multi-set work — all sets restored
        return {**out, **SESS.sets_info(), "n": len(SESS.instances), "nboxes": 0, "cached": "saved"}
    if _load_state(_save_dir()):                            # legacy single-set save — restored as set 1
        return {**out, **SESS.sets_info(), "n": len(SESS.instances), "nboxes": 0, "cached": "saved"}
    if auto_seed and _load_state(_cache_dir()):            # cached seed draft (only when auto-seed is on)
        return {**out, **SESS.sets_info(), "n": len(SESS.instances), "nboxes": 0, "cached": "draft"}
    nboxes = len(_load_seed_boxes(field, date, stem, W, H))
    return {**out, **SESS.sets_info(), "n": 0, "nboxes": nboxes, "cached": False}   # BLANK


def add_seeds():
    """APPEND the YOLO+SAM seed masks to the current instances. Never removes anything already there
    (your worked-on / saved masks stay). Uses the cached seed draft for speed if present, else runs SAM
    once and caches it (only when starting from a blank image, so the cache stays a pure seed draft)."""
    if SESS.img is None:
        return {"added": 0, "n": 0}
    H, W = SESS.img.shape[:2]
    was_empty = (len(SESS.instances) == 0)
    n0 = len(SESS.instances)
    cache_png = os.path.join(_cache_dir(), f"{SESS.stem}_instances.png")
    cache_js = os.path.join(_cache_dir(), f"{SESS.stem}_seed.json")

    if os.path.exists(cache_png) and os.path.exists(cache_js):     # append from cached seed draft (fast)
        lab = cv2.imread(cache_png, cv2.IMREAD_UNCHANGED)
        meta = json.load(open(cache_js))
        for inst in meta["instances"]:
            if "bbox" not in inst:
                continue
            x0, y0, x1, y1 = inst["bbox"]
            sub = np.ascontiguousarray(lab[y0:y1, x0:x1] == inst["id"])
            if sub.any():
                SESS.add({"bbox": [x0, y0, x1, y1], "sub": sub}, seed_box=inst["seed_box"])
    else:                                                          # no cache -> run SAM once
        boxes = _load_seed_boxes(SESS.field, SESS.date, SESS.stem, W, H)
        for box, m in zip(boxes, get_engine().seed_boxes(SESS.img, boxes)):
            sp = to_sparse(m)
            if sp is not None:
                SESS.add(sp, seed_box=[float(v) for v in box])
        if was_empty:                                             # only cache a pure (blank-start) seed draft
            _dump_state(_cache_dir())
    return {"added": len(SESS.instances) - n0, "n": len(SESS.instances)}


def _outline(over, sp, color, thick):
    """Draw an instance's contour on the BGRA overlay, working only inside its bbox (fast)."""
    x0, y0, x1, y1 = sp["bbox"]
    cnts, _ = cv2.findContours(sp["sub"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c + [x0, y0] for c in cnts]                   # shift bbox-local contour to full-image coords
    cv2.drawContours(over, cnts, -1, color, thick)


def _is_visible(it, hide_all):
    """An instance shows unless it's individually hidden, or hide-all is on and it isn't locked."""
    return (not it["hidden"]) and (it["locked"] or not hide_all)


def render_overlay(selected_id=None, hide_all=False, style="color", solid=False):
    """Composite instance masks into a BGRA PNG. Individually-hidden masks are skipped; with hide_all on,
    only LOCKED masks stay.
      style="color"  -> per-id colours (distinct wheat heads);  "binary" -> all masks white.
      solid=False    -> translucent (for viewing over the photo);  solid=True -> opaque (for the black/
                        white background modes, so every mask is fully visible like a mask image).
    In solid COLOUR mode each instance also gets a thin dark outline so touching heads stay separable."""
    if SESS.img is None:                              # overlay asked for before any image loaded
        ok, buf = cv2.imencode(".png", np.zeros((1, 1, 4), np.uint8))
        return buf.tobytes()
    H, W = SESS.img.shape[:2]
    over = np.zeros((H, W, 4), np.uint8)
    binary = (style == "binary")
    for it in SESS.instances:
        sp = it["mask"]
        if sp is None or not _is_visible(it, hide_all):
            continue
        x0, y0, x1, y1 = sp["bbox"]
        b, g, r = (255, 255, 255) if binary else color_for(it["id"])
        a = 255 if solid else (170 if it["id"] == selected_id else 90)
        over[y0:y1, x0:x1][sp["sub"]] = (b, g, r, a)
    if solid and not binary:                             # separate touching heads with a 1px dark border
        for it in SESS.instances:
            if it["mask"] is not None and _is_visible(it, hide_all):
                _outline(over, it["mask"], (0, 0, 0, 255), 1)
    if not binary:                                       # selection/lock cues only in colour mode (keep
        for it in SESS.instances:                        # the binary view a clean mask image)
            if it["mask"] is not None and it["locked"] and _is_visible(it, hide_all):
                _outline(over, it["mask"], (0, 215, 255, 255), 2)   # gold = locked
        it = SESS.find(selected_id) if selected_id else None
        if it and it["mask"] is not None:
            _outline(over, it["mask"], (255, 255, 255, 255), 3)     # white = selected
    ok, buf = cv2.imencode(".png", over)
    return buf.tobytes()


def hit_test(x, y):
    """Return the id of the smallest-area instance whose mask covers (x,y), or None (topmost pick)."""
    best, best_area = None, None
    for it in SESS.instances:
        sp = it["mask"]
        if sp is None:
            continue
        x0, y0, x1, y1 = sp["bbox"]
        if x0 <= x < x1 and y0 <= y < y1 and sp["sub"][int(y) - y0, int(x) - x0]:
            area = int(sp["sub"].sum())
            if best_area is None or area < best_area:
                best, best_area = it["id"], area
    return best


def refine_instance(it):
    """Re-run SAM on this instance's accumulated points (per-head crop). Keeps SAM2's 3 candidate masks on
    the instance (it["cands"]) so the UI can cycle them, and sets the mask to the auto-picked best."""
    pts = [[p[0], p[1]] for p in it["points"]]
    lbl = [p[2] for p in it["points"]]
    if not pts:
        it["mask"] = None
        it["cands"] = []; it["cand_idx"] = 0
        return 0
    masks, idx = get_engine().refine_points_all(SESS.img, pts, lbl, box_hint=it["seed_box"])
    it["cands"] = [to_sparse(m) for m in masks]      # transient (not saved to disk) — for cycling only
    it["cand_idx"] = idx
    it["mask"] = it["cands"][idx] if it["cands"] else None
    return int(it["mask"]["sub"].sum()) if it["mask"] else 0


def _rasterize_strokes(strokes, H, W):
    """Turn brush strokes (each = a list of image-space points + radius + erase flag) into two bool
    masks: pixels to ADD and pixels to ERASE. A stroke is drawn as round dabs at every point plus thick
    lines between them, so a fast drag still paints a continuous band."""
    add = np.zeros((H, W), np.uint8)
    ers = np.zeros((H, W), np.uint8)
    for s in strokes:
        tgt = ers if s.get("erase") else add
        r = max(1, int(round(s.get("r", 12))))
        pts = s.get("pts", [])
        for i, (px, py) in enumerate(pts):
            x, y = int(round(px)), int(round(py))
            cv2.circle(tgt, (x, y), r, 1, -1)                 # round cap / dab at each point
            if i > 0:                                         # thick line to the previous point
                x0, y0 = int(round(pts[i - 1][0])), int(round(pts[i - 1][1]))
                cv2.line(tgt, (x0, y0), (x, y), 1, 2 * r)
    return add.astype(bool), ers.astype(bool)


def apply_draw(it, add_bool, erase_bool):
    """Union ADD pixels into and subtract ERASE pixels from one instance's mask, then re-sparsify.
    This is how manual brush/polygon edits land in the SAME binary mask SAM would have produced."""
    H, W = SESS.img.shape[:2]
    full = np.zeros((H, W), bool)
    sp = it["mask"]
    if sp is not None:
        x0, y0, x1, y1 = sp["bbox"]
        full[y0:y1, x0:x1] = sp["sub"]
    if add_bool is not None:
        full |= add_bool
    if erase_bool is not None:
        full &= ~erase_bool
    it["mask"] = to_sparse(full)
    return int(full.sum())


def save_gt():
    """Write the finished GT. Three artifacts, all under manual_label/:
      - <stem>_sets/set<N>_instances.png (+ set<N>_seed.json + manifest.json) via _save_all_sets() —
        the uint16 INSTANCE MAP per set. This is the AUTHORITATIVE GT: read it via manifest.json's
        "active" set. (An older version also wrote a top-level <stem>_instances.png via _dump_state();
        it no longer does — any such leftover is STALE, see archive/gt_tool_stale_instances/.)
      - <stem>_gt_mask.png — the binary UNION of the active set (what eval_seg_2d reads).
      - <stem>_meta.json — count/size/backend.
    Saving also makes the image reload from disk on the next open."""
    if SESS.img is None:
        return {"error": "no image loaded"}
    H, W = SESS.img.shape[:2]
    out = _save_dir()
    os.makedirs(out, exist_ok=True)
    _save_all_sets()                                       # persist every set (resumable, incl. backups)

    union = np.zeros((H, W), np.uint8)
    n = 0
    for it in SESS.instances:
        sp = it["mask"]
        if sp is None:
            continue
        x0, y0, x1, y1 = sp["bbox"]
        union[y0:y1, x0:x1][sp["sub"]] = 255
        n += 1
    stem = SESS.stem
    cv2.imwrite(os.path.join(out, f"{stem}_gt_mask.png"), union)   # the eval_seg_2d artifact
    json.dump({"stem": stem, "count": n, "w": W, "h": H, "backend": os.path.basename(WEIGHT)},
              open(os.path.join(out, f"{stem}_meta.json"), "w"), indent=1)
    return {"count": n, "path": os.path.join(out, f"{stem}_gt_mask.png")}


def list_images():
    """The GT set from gt_selection.json + whether each already has a saved gt_mask."""
    sel = json.load(open(SELECTION))
    items = []
    for session, frames in sel.items():
        field, date = session.split("/")
        for fr in frames:
            stem = fr["stem"]
            done = os.path.exists(os.path.join(PHONE, field, date, "manual_label", f"{stem}_gt_mask.png"))
            cached = os.path.exists(os.path.join(PHONE, field, date, "gt_cache", f"{stem}_seed.json"))
            items.append({"field": field, "date": date, "stem": stem, "done": done, "cached": cached})
    return items


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: browser reuses one connection instead of churning many

    def log_message(self, *a):
        """Quiet the default per-request logging."""
        pass

    def _send(self, code, body, ctype="application/json"):
        """Write a response with the right headers. A dropped client connection (the browser aborting an
        image/overlay request when it gets superseded) is normal — swallow it instead of spewing a trace."""
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _body(self):
        """Parse the JSON request body into a dict."""
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _qs(self):
        """Parse the query string into a dict."""
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        return {k: v[0] for k, v in q.items()}

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/":
                html = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
                return self._send(200, html, "text/html; charset=utf-8")
            if path == "/api/list":
                return self._send(200, list_images())
            if path == "/image":
                q = self._qs()
                p = os.path.join(PHONE, q["field"], q["date"], "images", f"{q['stem']}.jpg")
                return self._send(200, open(p, "rb").read(), "image/jpeg")
            if path == "/api/overlay":
                q = self._qs()
                sel = int(q["sel"]) if q.get("sel", "") not in ("", "null") else None
                hide_all = q.get("hideall", "0") == "1"
                style = "binary" if q.get("style") == "binary" else "color"
                solid = q.get("solid", "0") == "1"
                with _lock:
                    png = render_overlay(sel, hide_all, style=style, solid=solid)
                return self._send(200, png, "image/png")
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            b = self._body()   # ALWAYS read the body (even if a handler ignores it) — otherwise an
            #                    unread body stays in the keep-alive buffer and corrupts the NEXT request
            with _lock:
                if path == "/api/load":
                    return self._send(200, load_image(b["field"], b["date"], b["stem"],
                                                      auto_seed=b.get("auto_seed", True)))
                if path == "/api/seed":            # append the YOLO+SAM seeds (auto on load, or the button)
                    return self._send(200, add_seeds())
                if path == "/api/select":
                    iid = hit_test(b["x"], b["y"])
                    it = SESS.find(iid) if iid else None
                    return self._send(200, {"id": iid, "locked": bool(it and it["locked"]),
                                            "points": it["points"] if it else [],
                                            "ncands": len(it.get("cands", [])) if it else 0,
                                            "cand_idx": it.get("cand_idx", 0) if it else 0})
                if path == "/api/set_points":
                    # commit an accumulated point set and run SAM once (the "Run" action)
                    it = SESS.find(b["id"])
                    if not it:
                        return self._send(404, {"error": "no instance"})
                    if it["locked"]:                        # locked = protected from edits
                        return self._send(200, {"id": it["id"], "locked": True})
                    it["points"] = [[float(p[0]), float(p[1]), int(p[2])] for p in b["points"]]
                    area = refine_instance(it)
                    return self._send(200, {"id": it["id"], "area": area, "npoints": len(it["points"]),
                                            "ncands": len(it.get("cands", [])), "cand_idx": it.get("cand_idx", 0)})
                if path == "/api/candidate":        # switch which of SAM's 3 candidate masks this head uses
                    it = SESS.find(b["id"])
                    if not it or it["locked"] or not it.get("cands"):
                        return self._send(200, {"ok": False})
                    n = len(it["cands"])
                    idx = int(b["idx"]) % n
                    it["cand_idx"] = idx
                    it["mask"] = it["cands"][idx]
                    area = int(it["mask"]["sub"].sum()) if it["mask"] else 0
                    return self._send(200, {"id": it["id"], "idx": idx, "n": n, "area": area})
                if path == "/api/flag":                     # toggle 'hidden' or 'locked' on one instance
                    it = SESS.find(b["id"])
                    if not it:
                        return self._send(404, {"error": "no instance"})
                    key = b["key"]
                    if key in ("hidden", "locked"):
                        it[key] = not it[key]
                    return self._send(200, {"id": it["id"], "hidden": it["hidden"], "locked": it["locked"]})
                if path == "/api/new":
                    # empty instance; the client places points then commits via /api/set_points
                    iid = SESS.add(None, seed_box=None, points=[])
                    return self._send(200, {"id": iid, "n": len(SESS.instances)})
                if path == "/api/undo_point":
                    it = SESS.find(b["id"])
                    if it and it["points"]:
                        it["points"].pop()
                        area = refine_instance(it)
                        if not it["points"]:      # no points left -> drop the instance
                            SESS.instances.remove(it)
                            return self._send(200, {"removed": True})
                        return self._send(200, {"id": it["id"], "area": area,
                                                "npoints": len(it["points"])})
                    return self._send(200, {"ok": True})
                if path == "/api/delete":
                    it = SESS.find(b["id"])
                    if it and it["locked"]:                 # locked = protected from delete
                        return self._send(200, {"locked": True, "n": len(SESS.instances)})
                    if it:
                        SESS.instances.remove(it)
                    return self._send(200, {"ok": True, "n": len(SESS.instances)})
                if path == "/api/set_add":          # create a new empty mask-set and make it active
                    SESS.add_set()
                    return self._send(200, {**SESS.sets_info(), "n": len(SESS.instances)})
                if path == "/api/set_switch":       # switch the active mask-set
                    name = b.get("name")
                    if name in SESS.sets:
                        SESS.active = name
                        SESS.backup = []
                    return self._send(200, {**SESS.sets_info(), "n": len(SESS.instances)})
                if path == "/api/clear":            # move masks into a NEW backup set (kept in the dropdown);
                    cleared = [it for it in SESS.instances if not it["locked"]]   # locked stay in place
                    kept = [it for it in SESS.instances if it["locked"]]
                    bname = None
                    if cleared:
                        src = SESS.active
                        bname = f"⟲ {src}"; k = 2
                        while bname in SESS.sets:
                            bname = f"⟲ {src} ({k})"; k += 1
                        SESS.sets[bname] = {"instances": cleared, "next_id": SESS.sets[src]["next_id"]}
                        SESS.sets[src]["instances"] = kept
                    return self._send(200, {**SESS.sets_info(), "n": len(SESS.instances), "backup_set": bname})
                if path == "/api/brush":
                    # paint/erase into the SELECTED head's mask (or a fresh head if none selected).
                    # Client accumulates strokes and only posts them here on commit (Enter / Create mask).
                    it = SESS.find(b["id"]) if b.get("id") else None
                    if it is None:                               # brushing with nothing selected -> new head
                        it = SESS.find(SESS.add(None, seed_box=None))
                    if it["locked"]:                            # locked = protected from edits
                        return self._send(200, {"id": it["id"], "locked": True})
                    add, ers = _rasterize_strokes(b.get("strokes", []), *SESS.img.shape[:2])
                    area = apply_draw(it, add, ers)
                    it["cands"] = []; it["cand_idx"] = 0        # manual edit -> SAM candidates are now stale
                    if it["mask"] is None:                      # erased down to nothing -> drop the head
                        SESS.instances.remove(it)
                        return self._send(200, {"removed": True, "n": len(SESS.instances)})
                    return self._send(200, {"id": it["id"], "area": area, "n": len(SESS.instances)})
                if path == "/api/polygon":
                    # fill a hand-drawn polygon into a brand-new head (draw a head SAM couldn't get)
                    verts = b.get("verts", [])
                    if len(verts) < 3:
                        return self._send(200, {"error": "need 3+ points"})
                    H, W = SESS.img.shape[:2]
                    m = np.zeros((H, W), np.uint8)
                    cv2.fillPoly(m, [np.array(verts, np.int32)], 1)
                    sp = to_sparse(m.astype(bool))
                    if sp is None:
                        return self._send(200, {"error": "empty polygon"})
                    iid = SESS.add(sp, seed_box=None)
                    return self._send(200, {"id": iid, "area": int(m.sum()), "n": len(SESS.instances)})
                if path == "/api/save":
                    return self._send(200, save_gt())
            return self._send(404, {"error": "not found"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._send(500, {"error": str(e)})


def main():
    """Start the threaded HTTP server."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[gt_tool] serving on http://localhost:{PORT}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gt_tool] bye.")


if __name__ == "__main__":
    main()
