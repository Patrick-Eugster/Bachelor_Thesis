"""Forced-centre CCT decode — the Phase 1 fix.

Stock CCTDecode runs its OWN blob search inside the image and keeps grabbing the
wrong blob (a code arc -> "7", the disk -> garbage). Here we instead TELL it where
the marker centre is, find the central dark disk right there, rectify the marker to a
frontal circle anchored on that disk, and read the code ring around it. No blob search
-> no arc artefacts.

We reuse CCTDecode's actual bit-reader (CCT_Decode) + validation (CCT_or_not) + affine
helper, so only the *candidate selection* changes. The code-ring sampling radius is
exposed (ring_scale) so we can calibrate it to the real markers later (stock = 2.5).

decode_at_center(bgr, cx, cy, ...) -> (code|None, info dict)
"""

import os
import sys
import math

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "cctdecode"))
import CCTDecodeRelease as _cct          # noqa: E402  (CCT_Decode, CCT_or_not)
from Support import my_getAffineTransform, PointAffineTransform  # noqa: E402


def find_disk_at(gray, cx, cy, search_r, cfg, debug=None):
    """Find the central dark disk near (cx,cy) via local Otsu, return its fitted
    ellipse (centre,(MA,ma),angle) in FULL-image coords, or None.

    Disk-vs-arc is decided by FILL RATIO = blob area / fitted-ellipse area: a solid central
    disk fills its ellipse (~1.0); a code arc is a thin sliver inside a big ellipse (<=~0.91).
    Fill is a SHAPE ratio, so it is SIZE-INDEPENDENT — no hardcoded pixel distance. The search
    window is the only size term and it scales with the candidate's radius (caller).
    If `debug` is a list, every considered blob is appended for visualisation."""
    H, W = gray.shape[:2]
    x0, y0 = max(0, int(cx - search_r)), max(0, int(cy - search_r))
    x1, y1 = min(W, int(cx + search_r)), min(H, int(cy + search_r))
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return None
    # local Otsu -> dark blobs become foreground
    _, bina = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_key = None, None
    for c in cnts:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < cfg["disk_min_area"]:
            continue
        (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
        if MA <= 0 or ma <= 0:
            continue
        circ = min(MA, ma) / max(MA, ma)                 # roundness in [0,1]
        hull = cv2.contourArea(cv2.convexHull(c))
        sol = area / hull if hull > 0 else 0
        fill = area / (math.pi / 4.0 * MA * ma)          # solid disk ~1.0, arc sliver << 1
        gx, gy = x0 + ex, y0 + ey
        off = math.hypot(gx - cx, gy - cy)
        passed = (off <= search_r and circ >= cfg["disk_min_circularity"]
                  and sol >= cfg["disk_min_solidity"] and fill >= cfg["disk_min_fill"])
        if debug is not None:
            debug.append({"center": (gx, gy), "axes": (MA, ma), "angle": ang,
                          "area": area, "fill": fill, "circ": circ, "sol": sol,
                          "off": off, "passed": passed})
        if not passed:
            continue
        # the central disk is the SOLID one (fill ~1); among solid blobs prefer the larger.
        # fill is the primary key (rounded so near-1.0 ties break on area, dropping specks).
        key = (round(fill, 2), area)
        if best is None or key > best_key:
            best_key = key
            best = ((gx, gy), (MA, ma), ang)
    return best


def find_center_concentric(gray, cx, cy, search_r, cfg, debug=None, _recenter=True):
    """Concentric-consensus disk finder (v8). The disk and every arc share ONE centre, so:
      * if a solid DISK is present (fill ~1) -> use it (sharpest centre);
      * else, reconstruct from the ARCS: every arc lies on the code ring, which is concentric
        with the disk, so fitting an ellipse to the combined arc points gives the RING ellipse,
        and the disk-equivalent ellipse = ring / ring_to_disk_ratio at the same centre.
    This recovers the centre when v6 lands on an arc AND when the disk itself is occluded, as
    long as >=2 arcs remain. Returns a disk-equivalent ellipse ((gx,gy),(MA,ma),ang) or None."""
    H, W = gray.shape[:2]
    x0, y0 = max(0, int(cx - search_r)), max(0, int(cy - search_r))
    x1, y1 = min(W, int(cx + search_r)), min(H, int(cy + search_r))
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return None
    _, bina = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    disks = []          # (fill, area, ellipse) for solid central-disk candidates
    arc_pts = []        # contour points of arc candidates (to fit the ring)
    for c in cnts:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if area < cfg["disk_min_area"]:
            continue
        (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
        if MA <= 0 or ma <= 0:
            continue
        circ = min(MA, ma) / max(MA, ma)
        hull = cv2.contourArea(cv2.convexHull(c))
        sol = area / hull if hull > 0 else 0
        fill = area / (math.pi / 4.0 * MA * ma)
        gx, gy = x0 + ex, y0 + ey
        if math.hypot(gx - cx, gy - cy) > search_r:
            continue
        if fill >= cfg["disk_min_fill"] and circ >= cfg["disk_min_circularity"] \
                and sol >= cfg["disk_min_solidity"]:
            kind = "disk"
            disks.append((round(fill, 2), area, ((gx, gy), (MA, ma), ang)))
        elif fill >= cfg.get("arc_min_fill", 0.30) and sol >= cfg.get("arc_min_solidity", 0.55):
            kind = "arc"
            arc_pts.append(c)            # window-local points; offset added at fit time
        else:
            kind = "reject"
        if debug is not None:
            debug.append({"center": (gx, gy), "axes": (MA, ma), "angle": ang,
                          "fill": fill, "sol": sol, "kind": kind})

    # determine a centre estimate: solid disk (sharpest) else reconstruct from the arc ring
    result = None
    if disks:
        disks.sort(reverse=True)         # by (fill, area)
        result = disks[0][2]
    else:
        ratio = cfg.get("ring_to_disk_ratio", 2.5)
        if len(arc_pts) >= 2:
            allpts = np.vstack(arc_pts).reshape(-1, 2).astype(np.float32)
            if len(allpts) >= 5:
                (rx, ry), (RMA, Rma), rang = cv2.fitEllipse(allpts)   # the code RING ellipse
                result = ((x0 + rx, y0 + ry), (RMA / ratio, Rma / ratio), rang)
                if debug is not None:
                    debug.append({"center": (x0 + rx, y0 + ry), "axes": (RMA, Rma),
                                  "angle": rang, "fill": -1, "sol": -1, "kind": "ring_fit"})
    if result is None:
        return None

    # RE-CENTRE once: if the centre is far from where we searched (v6 landed on an arc, or the
    # disk was clipped at the window edge), re-search centred on the estimate so the disk is now
    # fully inside the window and snaps cleanly. Guarded by _recenter to do this at most once.
    ecx, ecy = result[0]
    if _recenter and math.hypot(ecx - cx, ecy - cy) > cfg.get("recenter_frac", 0.35) * search_r:
        re = find_center_concentric(gray, ecx, ecy, search_r, cfg, debug, _recenter=False)
        if re is not None:
            return re
    return result


def decode_at_center(bgr, cx, cy, cfg, N=12, color="black", finder=None):
    """Decode the coded marker whose centre is at (cx,cy). Returns (code|None, info).
    `finder(gray,cx,cy,search_r,cfg)` locates the central disk (or a disk-equivalent ellipse
    reconstructed from the arcs); defaults to find_disk_at. v8 passes find_center_concentric."""
    if finder is None:
        finder = find_disk_at
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    info = {"disk": False, "valid_cct": False}

    disk = finder(gray, cx, cy, cfg["search_r"], cfg)
    if disk is None:
        return None, info
    info["disk"] = True
    (dx0, dy0), (MA, ma), ang = disk
    info["disk_radius"] = 0.25 * (MA + ma)
    # the centre of the disk we actually decode is the true fiducial centre — report it
    # (more accurate than the proposed centre, which can sit slightly off after NCC/snap)
    info["disk_center"] = [float(dx0), float(dy0)]

    # build the same concentric layout CCTDecode uses: box1=disk, box3=3x (marker edge)
    box1 = ((dx0, dy0), (MA, ma), ang)
    box3 = ((dx0, dy0), (MA * 3.0, ma * 3.0), ang)
    minRect = cv2.boxPoints(box3)
    a = max(box3[1][0], box3[1][1])
    s = a
    row_min, row_max = round(dy0 - s / 2), round(dy0 + s / 2)
    col_min, col_max = round(dx0 - s / 2), round(dx0 + s / 2)
    if not (row_min >= 0 and row_max <= H and col_min >= 0 and col_max <= W):
        return None, info                                 # marker ROI falls off-image
    roi = bgr[row_min:row_max, col_min:col_max]
    if roi.size == 0:
        return None, info

    # affine-rectify the (elliptical) marker ROI to a frontal circle (CCTDecode's method)
    ddx, ddy = dx0 - s / 2, dy0 - s / 2
    src = np.float32([[minRect[i][0] - ddx, minRect[i][1] - ddy] for i in range(4)]
                     + [[dx0 - ddx, dy0 - ddy]])
    dst = np.float32([[dx0 - a / 2 - ddx, dy0 - a / 2 - ddy],
                      [dx0 + a / 2 - ddx, dy0 - a / 2 - ddy],
                      [dx0 + a / 2 - ddx, dy0 + a / 2 - ddy],
                      [dx0 - a / 2 - ddx, dy0 + a / 2 - ddy],
                      [dx0 - ddx, dy0 - ddy]])
    M = my_getAffineTransform(src, dst)
    if isinstance(M, int):
        return None, info
    cct_img = cv2.warpAffine(roi, M, (round(s), round(s)))
    if cct_img.size == 0:
        return None, info
    cct_large = cv2.resize(cct_img, (0, 0), fx=200.0 / s, fy=200.0 / s,
                           interpolation=cv2.INTER_LANCZOS4)
    cct_gray = cv2.cvtColor(cct_large, cv2.COLOR_BGR2GRAY)
    _, cct_bina = cv2.threshold(cct_gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cct_eroded = cv2.erode(cct_bina, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # CCTDecode's own structural check. With the fill-ratio gate already guaranteeing a solid
    # central disk, this is largely redundant AND strict (it rejects tilted/edge markers whose
    # disk we DID find) — so it's optionally bypassable to recover recall (junk is still caught
    # by the fill gate + the degenerate-code filter downstream).
    info["valid_cct"] = bool(_cct.CCT_or_not(cct_eroded))
    if cfg.get("require_valid_cct", True) and not info["valid_cct"]:
        return None, info
    code = _cct.CCT_Decode(cct_eroded, N, color)
    return int(code), info


DEFAULT_CFG = {
    "search_r": 80,               # px window to look for the disk (caller scales it by radius)
    "disk_min_area": 25,
    "disk_min_circularity": 0.45,  # oblique markers are ellipses -> allow down to 0.45
    "disk_min_solidity": 0.85,
    "disk_min_fill": 0.92,         # area/ellipse-area: solid disk ~1.0, code arc <=~0.91 -> rejects arcs
    # --- v8 concentric finder: arc candidates + ring->disk geometry ---
    "arc_min_fill": 0.30,          # arc slivers: fill below the disk band but above noise
    "arc_min_solidity": 0.55,      # arcs are fairly solid crescents; below this = canopy clutter
    "ring_to_disk_ratio": 2.5,     # code ring radius / disk radius (CCTDecode geometry)
}
