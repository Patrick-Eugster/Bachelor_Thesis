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
WEIGHT = os.environ.get("GT_SAM_WEIGHT", os.path.join(REPO, "sam2.1_l.pt"))
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
    """Holds the one active image and its instances."""

    def __init__(self):
        self.field = self.date = self.stem = None
        self.img = None                 # RGB uint8 HxW3
        self.instances = []             # {id, seed_box, mask(sparse), points:[[x,y,l]]}
        self.next_id = 1
        self.pending_boxes = []         # seed boxes waiting for the (separate) seed pass

    def add(self, sparse, seed_box=None, points=None):
        """Register a new instance, return its id."""
        iid = self.next_id
        self.next_id += 1
        self.instances.append({"id": iid, "seed_box": seed_box, "mask": sparse,
                               "points": points or [], "hidden": False, "locked": False})
        return iid

    def find(self, iid):
        """Instance dict by id, or None."""
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


def load_image(field, date, stem):
    """Load the image, then restore instances from disk if we've seen this image before (saved GT first,
    then the seed cache) — that skips the ~30 s re-decode. Only a never-seen image needs a seed pass."""
    path = os.path.join(PHONE, field, date, "images", f"{stem}.jpg")
    img = np.array(Image.open(path).convert("RGB"))         # PIL = silent on Samsung JPEGs
    H, W = img.shape[:2]

    SESS.field, SESS.date, SESS.stem = field, date, stem
    SESS.img = img
    SESS.instances = []
    SESS.next_id = 1
    SESS.pending_boxes = []

    out = {"w": W, "h": H, "field": field, "date": date, "stem": stem, "roi": compute_roi_poly()}
    if _load_state(_save_dir()):                            # your saved corrections come back
        return {**out, "n": len(SESS.instances), "nboxes": 0, "cached": "saved"}
    if _load_state(_cache_dir()):                           # the cached draft (seeded before, not saved)
        return {**out, "n": len(SESS.instances), "nboxes": 0, "cached": "draft"}
    SESS.pending_boxes = _load_seed_boxes(field, date, stem, W, H)
    return {**out, "n": 0, "nboxes": len(SESS.pending_boxes), "cached": False}


def seed_current():
    """Run the SAM seed (only for a never-seen image), then CACHE the draft to disk so the next open of
    this image is instant."""
    if SESS.img is None:
        return {"n": 0}
    boxes = SESS.pending_boxes
    SESS.pending_boxes = []
    if boxes:
        masks = get_engine().seed_boxes(SESS.img, boxes)
        for box, m in zip(boxes, masks):
            sp = to_sparse(m)
            if sp is not None:
                SESS.add(sp, seed_box=[float(v) for v in box])
        _dump_state(_cache_dir())                          # cache the decode -> fast reopen
    return {"n": len(SESS.instances)}


def _outline(over, sp, color, thick):
    """Draw an instance's contour on the BGRA overlay, working only inside its bbox (fast)."""
    x0, y0, x1, y1 = sp["bbox"]
    cnts, _ = cv2.findContours(sp["sub"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c + [x0, y0] for c in cnts]                   # shift bbox-local contour to full-image coords
    cv2.drawContours(over, cnts, -1, color, thick)


def _is_visible(it, hide_all):
    """An instance shows unless it's individually hidden, or hide-all is on and it isn't locked."""
    return (not it["hidden"]) and (it["locked"] or not hide_all)


def render_overlay(selected_id=None, hide_all=False):
    """Composite instance masks into a BGRA PNG. Individually-hidden masks are skipped; with hide_all on,
    only LOCKED masks stay. Locked = gold outline, selected = white outline."""
    if SESS.img is None:                              # overlay asked for before any image loaded
        ok, buf = cv2.imencode(".png", np.zeros((1, 1, 4), np.uint8))
        return buf.tobytes()
    H, W = SESS.img.shape[:2]
    over = np.zeros((H, W, 4), np.uint8)
    for it in SESS.instances:
        sp = it["mask"]
        if sp is None or not _is_visible(it, hide_all):
            continue
        x0, y0, x1, y1 = sp["bbox"]
        b, g, r = color_for(it["id"])
        a = 170 if it["id"] == selected_id else 90
        over[y0:y1, x0:x1][sp["sub"]] = (b, g, r, a)
    for it in SESS.instances:                            # gold outline on locked (visible) instances
        if it["mask"] is not None and it["locked"] and _is_visible(it, hide_all):
            _outline(over, it["mask"], (0, 215, 255, 255), 2)
    it = SESS.find(selected_id) if selected_id else None   # white outline on the selected instance
    if it and it["mask"] is not None:
        _outline(over, it["mask"], (255, 255, 255, 255), 3)
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
    """Re-run SAM on this instance's accumulated points (per-head crop) and update its mask."""
    pts = [[p[0], p[1]] for p in it["points"]]
    lbl = [p[2] for p in it["points"]]
    if not pts:
        it["mask"] = None
        return 0
    m = get_engine().refine_points(SESS.img, pts, lbl, box_hint=it["seed_box"])
    it["mask"] = to_sparse(m)
    return int(m.sum())


def save_gt():
    """Write the finished GT: the resumable state (instance map + JSON, via _dump_state) PLUS the binary
    union mask that eval_seg_2d reads + a small meta. Saving here also makes the image reload from disk."""
    if SESS.img is None:
        return {"error": "no image loaded"}
    H, W = SESS.img.shape[:2]
    out = _save_dir()
    _dump_state(out)                                       # <stem>_instances.png + <stem>_seed.json (resumable)

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
                with _lock:
                    png = render_overlay(sel, hide_all)
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
            with _lock:
                if path == "/api/load":
                    b = self._body()
                    return self._send(200, load_image(b["field"], b["date"], b["stem"]))
                if path == "/api/seed":
                    return self._send(200, seed_current())
                if path == "/api/select":
                    b = self._body()
                    iid = hit_test(b["x"], b["y"])
                    it = SESS.find(iid) if iid else None
                    return self._send(200, {"id": iid, "locked": bool(it and it["locked"]),
                                            "points": it["points"] if it else []})
                if path == "/api/set_points":
                    # commit an accumulated point set and run SAM once (the "Run" action)
                    b = self._body()
                    it = SESS.find(b["id"])
                    if not it:
                        return self._send(404, {"error": "no instance"})
                    if it["locked"]:                        # locked = protected from edits
                        return self._send(200, {"id": it["id"], "locked": True})
                    it["points"] = [[float(p[0]), float(p[1]), int(p[2])] for p in b["points"]]
                    return self._send(200, {"id": it["id"], "area": refine_instance(it),
                                            "npoints": len(it["points"])})
                if path == "/api/flag":                     # toggle 'hidden' or 'locked' on one instance
                    b = self._body()
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
                    return self._send(200, {"id": iid})
                if path == "/api/undo_point":
                    b = self._body()
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
                    b = self._body()
                    it = SESS.find(b["id"])
                    if it and it["locked"]:                 # locked = protected from delete
                        return self._send(200, {"locked": True})
                    if it:
                        SESS.instances.remove(it)
                    return self._send(200, {"ok": True})
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
