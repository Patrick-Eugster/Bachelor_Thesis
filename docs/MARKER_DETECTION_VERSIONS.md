# Marker Localization — the four detector versions (v1 → v4)

How the Stage A marker **localizer** evolved across four versions, what changed in the
algorithm each time, and what each produced. Each version was built to fix the *specific
failure* of the previous one, so the clearest way to read this is top to bottom.

Scripts/configs:
- v1 — [`src/preprocessing/detect_markers.py`](../src/preprocessing/detect_markers.py)
- v2 — [`src/preprocessing/detect_markers_v2.py`](../src/preprocessing/detect_markers_v2.py)
- v3 — [`src/preprocessing/detect_markers_v3.py`](../src/preprocessing/detect_markers_v3.py)
- v4 — [`src/preprocessing/detect_markers_v4.py`](../src/preprocessing/detect_markers_v4.py)

Each writes overlays to its own folder (`marker_vis/`, `marker_vis_v2/`, `_v3/`, `_v4/`) and a
JSON to `logs/`, so all four can be compared on the same images. All are **localization only**
(no IDs, no triangulation) and **read-only** on the dataset. Benchmarks below are on
`field_A/20250609` (119 phone images, 6 physical markers).

---

## The shared problem and the marker's structure

The computer only sees a grid of pixels; we need rules that say "a marker is *here*." A marker is:

- a **white square plate** (the substrate),
- carrying a **black coded ring** — curved **arcs** that encode the ID,
- around a **central fiducial** = a **solid grey disk with a tiny white dot at its exact center**.

The **white dot is the precise reference point** — it is the surveyed point and the one
triangulation must use. Each version keys on a different part of this structure, which is why
their behavior differs so much.

---

## v1 — the white square

**Cue:** find the bright, **colorless square plate**.

**Algorithm:**
1. HSV threshold → keep **bright + low-saturation** pixels (white plate vs saturated green canopy).
2. Morphology **close then open** → fill the holes the black pattern punches, drop small specks.
3. Find contours → gate each on **area**, **aspect ratio** (square, not a streak), and
   **convex-hull solidity** (fills its bounding box).
4. Confirm a **dark blob inside** (the black pattern).
5. Center = centroid of the interior dark pixels.

**Result — bad.** "White square" is a weak cue: a sunny canopy has many bright, square-ish
patches → **many false positives** (≈108/image before tuning); **tilted/distant** plates fail the
square test → **misses**; and the center is a **fuzzy guess** (centroid of all dark pixels, not the
fiducial).

> **Drove v2:** the cue isn't distinctive and the center is imprecise.

---

## v2 — ellipses on the whole pattern

**Cue:** the marker is built from **circles**; a circle seen from any angle projects to an
**ellipse**. Detect ellipses → perspective-robust.

**Algorithm:**
1. **Canny edges** → outlines of everything.
2. **`cv2.fitEllipse`** on each edge contour → keep **round**, **well-fitting** ones (candidate
   ring/arc pieces).
3. **White-surround filter** → keep an ellipse only if it sits on **white** (a plate). This is the
   strong discriminator that drops the hundreds of canopy ellipses.
4. **Size-aware clustering** (union-find): one marker yields several ellipses (fiducial + arcs);
   merge ellipses that are close **relative to their size** into one marker. (A hardcoded pixel
   distance is wrong — a close plate's features are far apart in px, a distant plate's are near —
   so the merge radius scales with the ellipse size.)
5. Center = the cluster member with the **brightest center**.

**Result — finds plates well, wrong center.** Good recall and most canopy rejected, but the
**center sometimes lands on a coded arc** (step 5 guesses), and it **needs clustering** — which
first **split one plate into several rings** until the clustering was made size-aware.

> **Drove v3:** good at *finding* plates, bad at *pinpointing the center*.

---

## v3 — the central fiducial disk

**Cue:** detect the **central fiducial directly** — a **solid, round, dark disk with a white
dot**, on white. This *is* the precise point, and the structure is very specific.

**Algorithm:**
1. Dark threshold → dark blobs.
2. Keep only **solid + round** blobs (the fiducial disk; arcs are curved/elongated → rejected).
3. **White-surround** check (on a plate).
4. Center = centroid of the **white dot** inside the disk (sub-pixel — the real reference point).
5. **No clustering needed** — one fiducial per marker (so the v2 split bug cannot occur).

**Result — precise but low recall.** The center is **always correct** (the fiducial by
construction) and there are **near-zero false positives** (canopy has no such disk). But the
**fixed global dark threshold** only catches the disk under specific lighting/distance →
**recall collapsed** (58/119 frames found nothing). An adaptive threshold made it *worse*
(fragmented the disk, flooded the canopy with edges).

> **Drove v4:** perfect center + precision, but it can't *find* enough plates.

---

## v4 — the hybrid (current best)

**Cue:** use each of v2 and v3 for what it is good at — **v2 finds the plates; v3 pinpoints the
center inside each one.**

**Algorithm:**
1. **v2's method** → candidate marker **regions** (ellipses + white-surround + size-aware
   clustering). *(recall, false-positive rejection, no splits)*
2. For each region, run **v3's fiducial search locally** — a **local Otsu threshold** *inside that
   small window* (local, so robust — no fragile global value) → snap the center to the fiducial's
   white dot.
3. A region with **no fiducial inside → dropped** → extra false-positive rejection.

**Result — best heuristic version, but still not good enough.** **100% of detections are
fiducial-snapped** → center always on the middle (fixes #1 *mostly*); regions without a fiducial
are dropped → false positives largely gone (#2); recall sits between v2 and v3 (#3 improved — 17
zero-frames vs v3's 58). Slower (~2 min for 119, because of per-region local thresholding).
**Remaining problems (on close inspection):** recall is still poor (visible/partly-occluded
plates missed), occasional false positives survive (e.g. a ring on bare wheat), and the center
still sometimes drifts off the fiducial.

**Threshold hardening (contrast-relative).** The white tests were initially hardcoded
(`white V≥140`, `dot>150`), which is fragile to sunlight/shadow/phone/white-balance/JPEG. They
were made **contrast-relative**: "white" is now defined by a **local Otsu split** computed per
region (the bright/dark boundary the region's own histogram implies), not an absolute brightness.
Only the saturation ceiling (`S≤70`) stays absolute — "white = desaturated" is a chroma property,
fairly lighting-robust. No change in numbers on `field_A/20250609` (consistent lighting) — the
benefit is **cross-session robustness** (still to be validated on another session). The disk
detection already used local Otsu. ("Local Otsu" = run Otsu's automatic threshold *inside a small
window* so the dark/bright split adapts to that region's lighting, vs one global cutoff.)

---

## Side-by-side

| Version | Cue / theory | Center quality | False positives | Recall | Verdict |
|---|---|---|---|---|---|
| **v1** | white square plate | fuzzy (dark centroid) | many | misses tilted | bad |
| **v2** | ellipses (rings) | ⚠️ sometimes on an arc | few | good | finds plates, wrong center |
| **v3** | fiducial disk | ✅ exact | ✅ ~none | ❌ low | precise but can't find them |
| **v4** | v2 find + v3 center | ✅ exact (100%) | ✅ dropped | ⚠️ moderate | **best** |

Benchmarks on `field_A/20250609` (markers/image): v2 ≈ mean 2.1, v3 ≈ mean 0.8 (58 empty
frames), v4 ≈ mean 1.5 (17 empty frames, 100% fiducial-snapped).

---

## The throughline

- **v1 → v2:** swapped a weak cue (white square) for a geometric one (ellipses of the rings).
- **v2 → v3:** swapped "guess which piece is the center" for "detect the fiducial directly."
- **v3 → v4:** combined v2's *finding* with v3's *centering*.

The one thing no version fully cracks is **far-away plates**, where the fiducial is only a few
pixels wide and can't be confirmed as a solid disk with a white dot. But those same plates are
**large in the closer frames**, and triangulation only needs each marker localized in **≥2 views**
total — so per-frame recall does not have to be perfect.

---

## The heuristic ceiling — and what's next (Option A → Option B)

**Conclusion after v1–v4: hand-tuned heuristics have hit their ceiling.** Every cue we add (white,
ellipse, dark disk, contrast, Otsu) has *canopy lookalikes* that pass it and *marker exceptions*
(occluded/tilted/distant) that fail it. Tuning one knob trades recall for precision and back. On a
cluttered, variable canopy, no hand-crafted rule set gets high recall **and** high precision **and**
a correct center simultaneously. So the next step changes the *kind* of method, not the thresholds.

**Option A — template-match the central bullseye (no training; try first).** The fiducial (solid
dark disk + white center dot on white) is identical on every marker and **rotation-invariant**, so
slide a small bullseye template over the image at a few **scales** and take the correlation peaks.
Why it should beat the heuristics: it matches the *specific concentric pattern* (far more
discriminating than "roundish dark thing on something white-ish"), the **correlation peak is the
fiducial center by definition** (fixes the center drift), and a partly-occluded fiducial still
correlates. Caveats: distant fiducials are only a few px (hard); extreme perspective tilt turns the
disk into an ellipse and degrades the match (mild tilt is fine; rotation is free because circular).
Mechanistically this is the **simplest ancestor of a CNN** — a single, fixed, hand-set convolution
filter (cross-correlation with one template), with no learning, no depth, no nonlinearity.

**Option B — train a small learned detector (most robust; if A falls short).** Repurpose the repo's
existing **YOLO** pipeline: hand-label the 6 markers across ~30–40 images (bounded effort), train a
marker detector, then use the fiducial trick *inside* each detected box for the sub-pixel center. A
CNN is the **learned, deep, nonlinear generalization** of the same correlation idea — many filters
learned from data, stacked in layers, so it tolerates lighting/tilt/occlusion/blur that a single
rigid template can't. Cost: the manual labeling.

**Plan: prototype A, escalate to B if needed.** A is no-label and instant to test (slide the
template, overlay the peaks, look at whether the 6 fiducials light up vs canopy). If A's recall on
distant/occluded plates is insufficient, that failure is exactly the signal that we need the learned
version (B). v1–v4 scripts are kept for the record; A/B will be new versions.

See [`MARKER_DETECTION_STAGE_A.md`](MARKER_DETECTION_STAGE_A.md) for the detailed v1 spec, and
[`MARKER_INTEGRATION_PLAN.md`](MARKER_INTEGRATION_PLAN.md) for how localization feeds the rest of
the marker pipeline (IDs → triangulate → distances → compare to ground truth).
