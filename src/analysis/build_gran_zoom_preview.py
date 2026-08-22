"""Compose a FIP-7.3-style granularity preview from the plain colored SAM overlays (no TP/FP/FN) of the
three @4032 SAM2 granularities on the field_A/20250715 frame: full frame, per tile, per head. Each row
is one granularity, the full ROI overlay on the left with a dashed box marking the center zoom region,
and that region enlarged on the right. The same zoom box is used for all three rows, so the enlarged
panels show how the three granularities segment the SAME patch of heads. Read-only apart from the single
preview PNG it writes into the review folder.

    python src/analysis/build_gran_zoom_preview.py
"""
import os
import cv2
import numpy as np

REV = "docs/analysis_results/maskgen_phone_fig_review"
ROWS = [
    ("full frame", "figB_gran_1_fullframe_4032_sam2.jpg"),
    ("per tile",   "figB_gran_2_pertile_4032_sam2.jpg"),
    ("per head",   "figB_gran_3_perhead_4032_sam2.jpg"),
]
GREY = 114
TOL = 15
PAD = 25
ZOOM_FRAC = 0.30     # zoom box size as a fraction of the ROI-content box (smaller = stronger zoom)
ZOOM_DY = -150       # shift the zoom center up (negative) / down by this many original pixels
PANEL_W = 1200       # width each of the two panels is scaled to
SEP = 12             # white separator thickness
OUT = os.path.join(REV, "gran_zoom_preview.png")
OUT_THESIS = "thesis/figures/maskgen_phone_gran_zoom.jpg"
THESIS_W = 1900      # width cap for the compressed thesis copy


def content_bbox(im):
    """Bounding box (x0,y0,x1,y1) of the non-grey ROI content, padded a little."""
    content = (np.abs(im.astype(np.int16) - GREY) > TOL).any(axis=2)
    ys, xs = np.where(content)
    return (max(0, xs.min() - PAD), max(0, ys.min() - PAD),
            min(im.shape[1], xs.max() + PAD), min(im.shape[0], ys.max() + PAD))


def dashed_rect(img, x0, y0, x1, y1, color=(0, 0, 0), t=6, dash=34, gap=24):
    """Draw a dashed rectangle in place (segments along each side)."""
    def line(p0, p1):
        p0 = np.array(p0, float); p1 = np.array(p1, float)
        L = np.hypot(*(p1 - p0)); d = (p1 - p0) / L
        s = 0.0
        while s < L:
            a = p0 + d * s; b = p0 + d * min(s + dash, L)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, t, cv2.LINE_AA)
            s += dash + gap
    line((x0, y0), (x1, y0)); line((x1, y0), (x1, y1))
    line((x1, y1), (x0, y1)); line((x0, y1), (x0, y0))


def label(img, text):
    """White-boxed label in the top-left corner."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
    cv2.rectangle(img, (16, 16), (16 + w + 24, 16 + h + 28), (255, 255, 255), -1)
    cv2.putText(img, text, (28, 16 + h + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4, cv2.LINE_AA)


def scale_to_w(im, w):
    """Resize to width w, keeping aspect."""
    h = round(im.shape[0] * w / im.shape[1])
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)


def main():
    # the three overlays share the frame + ROI, so fix the content box and the zoom box once
    ref = cv2.imread(os.path.join(REV, ROWS[0][1]))
    cx0, cy0, cx1, cy1 = content_bbox(ref)
    cw, ch = cx1 - cx0, cy1 - cy0
    zw, zh = int(cw * ZOOM_FRAC), int(ch * ZOOM_FRAC)
    mx, my = (cx0 + cx1) // 2, (cy0 + cy1) // 2 + ZOOM_DY
    zx0, zy0 = mx - zw // 2, my - zh // 2
    zx1, zy1 = zx0 + zw, zy0 + zh

    rows_img = []
    for name, fname in ROWS:
        im = cv2.imread(os.path.join(REV, fname))
        full = im[cy0:cy1, cx0:cx1].copy()
        # dashed box in the coordinates of the cropped full panel
        dashed_rect(full, zx0 - cx0, zy0 - cy0, zx1 - cx0, zy1 - cy0)
        label(full, name)
        zoom = im[zy0:zy1, zx0:zx1].copy()
        left = scale_to_w(full, PANEL_W)
        right = scale_to_w(zoom, PANEL_W)
        # match heights so they sit side by side (pad the shorter with white)
        H = max(left.shape[0], right.shape[0])
        def padH(a):
            if a.shape[0] == H:
                return a
            pad = np.full((H - a.shape[0], a.shape[1], 3), 255, np.uint8)
            return np.vstack([a, pad])
        sep = np.full((H, SEP, 3), 255, np.uint8)
        rows_img.append(np.hstack([padH(left), sep, padH(right)]))

    Wr = max(r.shape[1] for r in rows_img)
    hsep = np.full((SEP, Wr, 3), 255, np.uint8)
    stacked = []
    for i, r in enumerate(rows_img):
        if r.shape[1] < Wr:
            r = np.hstack([r, np.full((r.shape[0], Wr - r.shape[1], 3), 255, np.uint8)])
        stacked.append(r)
        if i < len(rows_img) - 1:
            stacked.append(hsep)
    composed = np.vstack(stacked)
    cv2.imwrite(OUT, composed)
    print(f"wrote {OUT}  ({composed.shape[1]}x{composed.shape[0]})")
    # compressed copy for the thesis (downscaled + jpg so it does not bloat the PDF)
    thesis = composed
    if thesis.shape[1] > THESIS_W:
        h = round(thesis.shape[0] * THESIS_W / thesis.shape[1])
        thesis = cv2.resize(thesis, (THESIS_W, h), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(OUT_THESIS), exist_ok=True)
    cv2.imwrite(OUT_THESIS, thesis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {OUT_THESIS}  ({thesis.shape[1]}x{thesis.shape[0]})")
    print(f"zoom box (orig px): x {zx0}-{zx1}, y {zy0}-{zy1}")


if __name__ == "__main__":
    main()
