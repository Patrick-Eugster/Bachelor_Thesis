# Marker Localization — Stage A

How the Stage A marker **localizer** works
([`src/preprocessing/detect_markers.py`](../src/preprocessing/detect_markers.py),
config [`configs/preprocessing/detect_markers.yaml`](../configs/preprocessing/detect_markers.yaml)).

**What Stage A does:** given one undistorted photo, find the pixel location of every visible
coded-marker plate. **What it does not do:** read which marker it is (that's Stage B, template
matching) or compute 3D positions (Step 2, triangulation). It only writes overlay PNGs to
`marker_vis/` and a detections JSON to `logs/` — the dataset itself is never modified.

The markers are **coded targets**: a white square plate with a black circular disk in the
middle (the disk carries a coded ring + a white center dot that encode the ID). Against a green
wheat canopy the useful signal is simple — the plate is **bright and white (colorless)**, the
canopy is **green (colorful)**. The whole detector is built around exploiting that one contrast,
followed by a few sanity checks. It is a classical computer-vision pipeline (color + shape
heuristics), not a trained neural network — which is appropriate because we only have 6 fixed
targets and no labeled training data.

---

## The pipeline, step by step

```
photo ──▶ 1. color filter ──▶ 2. clean-up ──▶ 3. find shapes ──▶ 4. shape checks ──▶ 5. "has a disk?" ──▶ centers
            (HSV threshold)    (morphology)     (contours)         (size/square)        (dark fraction)
```

Each step is a cheap filter that throws away non-markers; what survives all five is reported as
a marker. Every threshold mentioned below lives in the YAML config and was set by *measuring real
plates* (see "Calibration" at the end), not guessed.

### 1. Color filter — keep bright, colorless pixels

We convert the image from RGB to **HSV** (Hue, Saturation, Value). The reason: HSV separates
*how colorful* a pixel is (Saturation) and *how bright* it is (Value) into their own channels,
so "bright and colorless" becomes a simple two-condition test. In RGB those two properties are
tangled across all three channels and much harder to threshold.

We then build a black-and-white **mask**: a pixel is kept (white in the mask) only if

- its **Value ≥ 150** → it is bright, *and*
- its **Saturation ≤ 60** → it is colorless (white/grey), not saturated green.

The saturation condition is the one doing the real work, because the canopy is also fairly
bright — it's the *greenness* that separates it from the plate. Hue is ignored on purpose (white
has no meaningful hue). The result is a mask where each plate shows up as a white square — but
with a hole in the middle, because the black disk is dark and fails the brightness test. The mask
is also speckled with small stray bright spots (sky glints between wheat heads).

*(Code: `white_plate_mask()`.)*

### 2. Clean-up — fill the holes, remove the specks

The raw mask has two problems we fix with **morphology** (standard image operations that grow or
shrink white regions using a small circular brush called a *structuring element*):

- **Closing** (grow then shrink, brush size 15 px): fills the hole left by the black disk so each
  plate becomes one **solid** white square. This matters because the next checks assume a filled
  shape.
- **Opening** (shrink then grow, brush size 5 px): deletes white specks too small to survive the
  shrink — the sky-glint noise disappears, the real plates stay.

Order matters: we close first (make plates solid), then open (remove noise). After this we have a
clean mask: a handful of solid white squares.

*(Code: `clean_mask()`. The brush sizes are tuned for ~4000-px-wide phone photos.)*

### 3. Find the shapes

`cv2.findContours` walks the mask and returns the **outline** of each separate white region — one
polygon per connected blob. From here on we examine each region individually and decide whether it
looks like a marker. (We use only the outer outline since the holes are already filled.)

### 4. Shape checks — is this region plate-shaped?

For each region we compute three descriptors and reject anything outside the expected range:

- **Size** — the region's area as a *fraction of the whole image* (so the threshold is independent
  of resolution). Real plates measured between ~0.0015 and ~0.01 of the frame. The lower bound is
  what kills most false positives (tiny bright canopy gaps); it does cost us the most distant,
  small plates, but that's fine — each marker only needs to be found clearly in **2 photos** to be
  located in 3D later, and every marker is large in *some* views.
- **Squareness (aspect ratio)** — we fit the tightest *rotated* rectangle around the region and
  compare its longer side to its shorter side. A real plate, even tilted, stays roughly square
  (ratio ≤ 2). Long thin things (wheat rows, leaf-edge smears) get a high ratio and are rejected.
- **Solidity** — how completely the shape fills that rotated rectangle. A real square plate fills
  most of it; a ragged sliver doesn't. **Important detail:** we measure the fill using the
  region's **convex hull** (imagine shrink-wrapping a rubber band around it), *not* the raw region.
  This is because the black disk often touches the plate's edge, breaking the white border into a
  ragged, concave shape — the raw fill ratio of a genuine plate comes out low (~0.3–0.5) and would
  be wrongly rejected. The convex hull restores a clean square outline, so real plates score ~0.8
  and pass. **Switching this one check from raw-area to hull-area is the fix that made the detector
  work** (it went from rejecting everything, to ~the right count).

*(Code: `detect_one()` using `cv2.contourArea`, `cv2.minAreaRect`, `cv2.convexHull`.)*

### 5. "Has a black disk?" — the decisive check

A bright square patch of field could still slip through the shape checks. The clincher is that a
real marker has a **black disk inside it**. Within each surviving region's bounding box we count
the fraction of **dark** pixels (Value < 80) and require it to land in a sensible band (between
0.12 and 0.5):

- too few dark pixels → it's a blank bright patch, not a marker → reject;
- almost all dark → it's a shadow, not a white plate → reject.

This appearance check is the single strongest discriminator, and we run it last because it's the
most expensive (it inspects the pixels inside each candidate).

*(Code: `dark_center()`.)*

### Reporting the center

For everything that passes, the reported marker location is the **centroid of those dark pixels**
(i.e. the center of the black disk) — a more stable, code-centered point than the middle of the
white square. This is a Stage-A approximation; the exact sub-pixel target point (the tiny white
center dot) is recovered in Stage B once the marker is identified and rectified.

---

## Output

For each detected marker we save its center, bounding box, and the four scores it passed on
(size, aspect, solidity, dark-fraction) — keeping the scores means any accept/reject decision can
be checked afterwards from `logs/marker_detections.json`. We also draw the box + center dot onto a
down-scaled copy of the photo in `marker_vis/`, and the run prints a summary (how many candidates
per image, how many images hit the expected 6, how many found nothing).

---

## What it gets right and wrong

| Behavior | Cause | Why it's acceptable |
|---|---|---|
| Finds the prominent plates reliably | strong white-vs-green contrast | this is the main job |
| Misses heavily-tilted or distant plates | fails the size or squareness check in that view | the marker is still found in its closer/frontal views; 2 views are enough |
| Occasional false positive on bare canopy | a bright, square, shadowed patch passes all checks | **Stage B rejects it** (it matches none of the 6 known codes), and it won't triangulate consistently |

Stage A is deliberately tuned for **good-enough recall, moderate precision** — its job is to
*propose* candidate plate locations. Assigning IDs and removing false positives is Stage B's job.

Performance: about 0.27 s per image (119 images in ~32 s, single-threaded), mostly JPEG decoding
and the color conversion.

---

## Parameters (all in the YAML)

| Setting | Default | What it controls |
|---|---|---|
| `v_min`, `s_max` | 150, 60 | the bright + colorless test for the white mask (step 1) |
| `close_kernel`, `open_kernel` | 15, 5 | hole-fill and despeckle brush sizes (step 2) |
| `min_area_frac`, `max_area_frac` | 0.0015, 0.01 | allowed plate size as a fraction of the image (step 4) |
| `aspect_max` | 2.0 | max long/short side ratio — squareness (step 4) |
| `min_extent` | 0.7 | min convex-hull fill of the rotated rect — solidity (step 4) |
| `dark_v_max` | 80 | brightness below which a pixel counts as "dark disk" (step 5) |
| `dark_frac_min`, `dark_frac_max` | 0.12, 0.5 | allowed fraction of dark pixels inside a plate (step 5) |

---

## Calibration (how the thresholds were set)

The thresholds aren't arbitrary — they came from measuring the actual plates on a representative
frame of `field_A/20250609`. Real plates clustered at: size 0.002–0.006 of the frame, aspect
1.0–1.95, **convex-hull** solidity 0.71–0.89 (but raw solidity only 0.27–0.56 — the reason for the
hull fix), and dark-fraction 0.17–0.37. Each gate was placed just outside these measured ranges.

The effect of getting this right:

| Configuration | candidates per image | verdict |
|---|---|---|
| loose gates, raw-area solidity | ~107 | unusable — every canopy speck passes |
| raw-area solidity at 0.7 | 0 | over-tight — real plates fail raw solidity |
| **convex-hull solidity + calibrated gates** | **~3–6 (median 3 over 119 images)** | matches the 6 physical markers |

---

## Where this fits

Stage A (find the plates) → **Stage B** (template-match the 6 known code images to label each one
`target 1…6` and drop false positives) → **Step 2** (triangulate the labeled points with our
COLMAP camera poses to get 3D marker positions) → **Step 3** (compute marker-to-marker distances)
→ **Step 4** (compare those distances to the surveyed ground truth and to Agisoft's ~5–15 mm).
The full plan and the phone-vs-FIP data map are in
[`MARKER_INTEGRATION_PLAN.md`](MARKER_INTEGRATION_PLAN.md).
