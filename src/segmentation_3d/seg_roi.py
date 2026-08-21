"""
seg_roi.py — restrict the 3D segmentation to the plot region of interest (ROI).

This is the 3D counterpart of the mask-generation ROI in
`src/mask_generation/roi_mask.py`. That one works in 2D: it greys out the image
outside the marker hull before YOLO/SAM. This one works in 3D: it marks which
Gaussians of the trained model lie inside the plot ROI, so the segmentation
never gathers the ground and canopy BEHIND the plot into one large background
blob (and, optionally, keeps the coded-marker stakes out of the head set).

Both use the SAME source of truth: the 6 coded ground markers, already
triangulated to 3D in `logs/marker_points3d.json` (in the sparse/0 frame the
Gaussians are trained in). Here we:
  1. fit a plane through the 6 markers (their ground plane — the plot is flat,
     max fit residual ~2 cm),
  2. take the convex hull of the markers IN that plane -> the plot ROI polygon,
  3. keep a Gaussian if its in-plane position is inside the ROI hull (grown
     outward by roi_buffer_m so heads on the boundary are not clipped) AND its
     height above the plane is within the canopy band.

F2 (marker exclusion) additionally drops Gaussians within marker_radius_m of any
marker centre in the plane — a marker stake is a vertical line through the
marker, so in the plane it is just that point and the exclusion is an in-plane
radius.

ALL-MARKERS GATE (same policy as roi_mask): the hull needs the full marker ring.
If COLMAP failed to triangulate even one manifest marker, a partial hull would
cave in on the missing side and clip real heads, so we WARN and return None
(ROI disabled -> caller keeps its previous behaviour). OFF by default at the
call site, so a normal seg run is byte-identical.
"""

import os
import json

import numpy as np
import torch


def _load_markers(plot_dir):
    """Load the triangulated 3D markers for one plot. Returns an (N,3) array of the
    solved marker centres, or None if the file is missing or any manifest marker is
    unsolved (all-markers gate — a partial ring would clip real heads)."""
    mk_path = os.path.join(plot_dir, "logs", "marker_points3d.json")
    if not os.path.isfile(mk_path):
        print(f"WARNING [seg_roi]: no marker file at {mk_path} — ROI disabled (no cull).")
        return None
    with open(mk_path) as f:
        mk = json.load(f)
    pts3d = mk.get("points3d", {})
    solved = {code: v for code, v in pts3d.items() if v}          # drop null / unsolved markers
    manifest = mk.get("manifest") or list(pts3d.keys())
    if len(solved) < len(manifest):
        have = {str(k) for k in solved}
        missing = [c for c in manifest if str(c) not in have]
        print(f"WARNING [seg_roi]: only {len(solved)}/{len(manifest)} markers triangulated "
              f"(missing {missing}) — ROI disabled for this plot; a partial hull would clip heads.")
        return None
    return np.array([v["xyz"] for v in solved.values()], dtype=np.float64)


def _fit_plane(markers, all_xyz):
    """Fit the marker ground plane. Returns (centroid c, in-plane basis u,v, up-normal n).
    n is oriented so the bulk of the model's Gaussians sit on the +n (above-ground) side,
    since the wheat heads and canopy are above the marker plane."""
    c = markers.mean(axis=0)
    _, _, Vt = np.linalg.svd(markers - c)
    u, v, n = Vt[0], Vt[1], Vt[2]                                  # n = smallest-variance dir = plane normal
    # orient n toward where the Gaussians are (heads are above the ground plane)
    if np.dot(all_xyz.mean(axis=0) - c, n) < 0:
        n = -n
    return c, u, v, n


def _inside_convex_hull(uv, hull_uv, buffer):
    """Vectorised point-in-convex-polygon with an outward buffer. hull_uv is the ordered
    convex-hull polygon (K,2); a point is kept if it is on the inside side of every edge,
    allowing a slack of `buffer` (so the polygon is grown outward by buffer, mitred corners).
    Returns a boolean array over uv rows."""
    K = len(hull_uv)
    # signed area to get consistent winding (so "inside" is a single sign)
    area2 = 0.0
    for i in range(K):
        x0, y0 = hull_uv[i]
        x1, y1 = hull_uv[(i + 1) % K]
        area2 += x0 * y1 - x1 * y0
    orient = 1.0 if area2 > 0 else -1.0                            # +1 CCW, -1 CW
    keep = np.ones(len(uv), dtype=bool)
    for i in range(K):
        a = hull_uv[i]
        b = hull_uv[(i + 1) % K]
        e = b - a
        elen = np.hypot(e[0], e[1]) + 1e-12
        # signed distance of each point to the left of edge a->b (positive = inside for CCW)
        cross = (e[0] * (uv[:, 1] - a[1]) - e[1] * (uv[:, 0] - a[0])) / elen
        keep &= (orient * cross) >= -buffer                        # inside, grown outward by buffer
    return keep


def _convex_hull_2d(pts):
    """Andrew's monotone-chain convex hull of 2D points -> ordered hull vertices (K,2), CCW.
    Small pure-numpy hull so this module doesn't need cv2 (kept import-light for the train env)."""
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(P) <= 2:
        return P

    def _half(points):
        h = []
        for p in points:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    lower = _half(P)
    upper = _half(P[::-1])
    return np.array(lower + upper)


def default_ground_filter(gaussian_xyz, plot_dir, ground_percentile=10.0, verbose=True):
    """Gentle, scale-free, tilt-correct ground cull that KEEPS HEADS WHOLE.

    Replaces the crude z<z_mean. That heuristic culled the lower HALF of the scene by height, which only
    works when the heads are the topmost layer (true for the FIP overhead rig). In phone capture the
    heads sit at MID-height, so a mean-height cut (world-z OR plane-relative) slices through every head:
    each head loses its lower part, renders as a partial/smaller blob, matches poorly across views, and
    the segmentation crawls (masks never get consumed).

    Instead we fit the marker plane (the plot's true 'up', any orientation) and cull only the lowest
    `ground_percentile` % of Gaussians BY HEIGHT above that plane — i.e. a thin slice of clearly
    below-ground junk — so the whole canopy incl. every head is kept. Using a PERCENTILE makes it
    scale-free, so the same setting works for the arbitrary-scale COLMAP frame and the metric Agisoft
    frame. Applies whenever the plot has markers, tilted or level (a mean cut bisects heads either way).

    Returns a boolean torch tensor (True = cull), or None if markers are unavailable — then the caller
    falls back to the legacy z<z_mean, which is correct for FIP (no markers, heads ARE the top layer).
    Independent of roi_cull."""
    xyz = gaussian_xyz.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(gaussian_xyz) \
        else np.asarray(gaussian_xyz, dtype=np.float64)
    markers = _load_markers(plot_dir)
    if markers is None:
        return None
    c, u, v, n = _fit_plane(markers, xyz)
    h = (xyz - c) @ n                                   # height along the true up-axis
    thr = np.percentile(h, ground_percentile)          # cut a thin below-ground slice, scale-free
    cull = h < thr
    if verbose:
        tilt = np.degrees(np.arccos(abs(n @ np.array([0.0, 0.0, 1.0]))))
        print(f"[seg_roi] ground cull: marker plane tilted {tilt:.1f} deg from world-z; culling the "
              f"bottom {ground_percentile:.0f}% by height (h < {thr:.3f}) = {int(cull.sum())}/{len(xyz)} "
              f"Gaussians ({100.0 * cull.mean():.1f}%) — thin below-ground slice, heads kept whole")
    return torch.from_numpy(cull)


def build_roi_keep_mask(gaussian_xyz, plot_dir,
                        roi_cull=True, height_band=False, marker_exclude=False,
                        roi_buffer_m=0.25, band_m=(-0.50, 1.50),
                        marker_radius_m=0.075, marker_radius_rel=0.0, verbose=True):
    """Boolean keep-mask over the Gaussians (True = keep). Three INDEPENDENT filters that AND together
    — enable any combination:

    roi_cull       (F1, the ROI): keep only Gaussians whose horizontal position is INSIDE the
                   marker-hull plot region (grown outward by roi_buffer_m). This is the plot ROI.
    height_band    (separate vertical filter, NOT part of the ROI): keep only Gaussians whose height
                   above the marker plane is within band_m — drops sky floaters / deep-underground
                   junk that sit over the plot. Off by default; near a no-op on flat wheat plots.
    marker_exclude (F2): drop Gaussians in a small 3D sphere (radius marker_radius_m) around each
                   coded-marker centre — the plate itself, not a vertical column, so nearby heads live.

    gaussian_xyz : (G,3) tensor/array of Gaussian centres, in the sparse/0 world frame.
    plot_dir     : the plot's source folder (holds logs/marker_points3d.json + sparse/).
    roi_buffer_m : grow the ROI hull outward this many metres so plot-edge heads aren't clipped.
    band_m       : (low, high) metres relative to the marker plane, used only when height_band=True.
    marker_radius_m : 3D-sphere radius around each marker centre (plates are ~13 cm circle / 15 cm
                      square). NOTE the RECONSTRUCTED marker is ~3x the plate (3DGS also bakes the plate's
                      shadow + a fuzzy halo, measured ~0.20-0.25 m radius on A/0715), so 0.075 only carves
                      the core and the coloured rim leaks into neighbour heads — prefer marker_radius_rel.
    marker_radius_rel : if > 0, OVERRIDE marker_radius_m with (marker_radius_rel * median pairwise marker
                      spacing). This is SCALE-FREE — COLMAP's world units are arbitrary per session, so a
                      fixed metre radius silently means different physical sizes across sessions; a fraction
                      of the marker spacing tracks the real geometry. ~0.065 reproduces the 0.20 m we want
                      on A/0715 (median spacing ~3.1 u). Default 0.0 = off = use the absolute marker_radius_m.

    Returns a boolean torch tensor (G,), or None if nothing is enabled or the markers can't be built
    (missing/partial ring) — the caller then keeps its previous behaviour (byte-identical).
    """
    if not (roi_cull or height_band or marker_exclude):
        return None
    xyz = gaussian_xyz.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(gaussian_xyz) \
        else np.asarray(gaussian_xyz, dtype=np.float64)

    markers = _load_markers(plot_dir)
    if markers is None:
        return None

    c, u, v, n = _fit_plane(markers, xyz)
    basis = np.stack([u, v], axis=1)                              # (3,2): world -> in-plane (u,v)
    d = xyz - c
    uv = d @ basis                                                # (G,2) in-plane coords
    height = d @ n                                                # (G,) signed height above plane

    mk_uv = (markers - c) @ basis                                # (M,2) markers in-plane

    keep = np.ones(len(xyz), dtype=bool)
    if roi_cull:
        # F1 = the ROI: horizontal in/out of the marker hull (grown outward by roi_buffer_m)
        hull = _convex_hull_2d(mk_uv)
        keep &= _inside_convex_hull(uv, hull, roi_buffer_m)

    if height_band:
        # separate vertical filter (NOT the ROI): keep only heights within band_m of the plane
        keep &= (height >= band_m[0]) & (height <= band_m[1])

    if marker_exclude:
        # CONSERVATIVE: exclude only a 3D SPHERE around each marker centre — the coded marker as
        # reconstructed (plate + shadow + halo) — NOT the whole vertical column. An in-plane (column)
        # test would cull heads above/beside a marker; a sphere removes just the marker blob so heads a
        # bit away in any direction survive. Uses true 3D distance to the marker xyz.
        if marker_radius_rel > 0 and len(markers) >= 2:
            # scale-free radius: a fraction of the median pairwise marker spacing (COLMAP units are
            # arbitrary per session, so a fixed metre value is unsafe across sessions).
            dmat = np.linalg.norm(markers[:, None, :] - markers[None, :, :], axis=2)
            spacing = np.median(dmat[np.triu_indices(len(markers), k=1)])
            radius = marker_radius_rel * spacing
        else:
            radius = marker_radius_m
        near = np.zeros(len(xyz), dtype=bool)
        for m in markers:
            near |= (np.linalg.norm(xyz - m, axis=1) <= radius)
        keep &= ~near

    if verbose:
        g = len(xyz)
        mr = (f"{radius:.3f}u (rel {marker_radius_rel}xspacing)" if (marker_exclude and marker_radius_rel > 0
                                                                     and len(markers) >= 2)
              else f"{marker_radius_m}m")
        print(f"[seg_roi] keep {int(keep.sum())}/{g} Gaussians "
              f"({100.0 * keep.sum() / max(g, 1):.1f}%) — culled {g - int(keep.sum())} "
              f"(roi_cull={roi_cull} buffer={roi_buffer_m}m, height_band={height_band} band={band_m}m, "
              f"marker_exclude={marker_exclude} marker_r={mr})")
    return torch.from_numpy(keep)
