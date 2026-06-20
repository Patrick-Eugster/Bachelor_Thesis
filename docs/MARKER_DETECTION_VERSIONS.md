# Marker Localization — the detector versions (v1 → v6)

How the Stage A marker **localizer** evolved across six versions, what changed in the
algorithm each time, and what each produced. Each version was built to fix the *specific
failure* of the previous one, so the clearest way to read this is top to bottom.

**Bottom line up front:** v1–v4 were hand-tuned classical-CV heuristics (color/shape/contrast
rules). v5–v6 switched to **template matching** (Option A): match a picture of the fiducial against
the image. v6 (a **real** cropped fiducial + multi-scale NCC + fiducial-snap + ellipse-fit center +
dedup) is clearly the best version — but it **still isn't good enough** (on cluttered frames only
~1 of 5 proposed markers is correct). Conclusion: **template-matching has also hit a ceiling →
next is Option B, a trained CNN/YOLO marker detector** (see the last section).

Scripts/configs:
- v1 — [`src/preprocessing/detect_markers.py`](../src/preprocessing/detect_markers.py)
- v2 — [`src/preprocessing/detect_markers_v2.py`](../src/preprocessing/detect_markers_v2.py)
- v3 — [`src/preprocessing/detect_markers_v3.py`](../src/preprocessing/detect_markers_v3.py)
- v4 — [`src/preprocessing/detect_markers_v4.py`](../src/preprocessing/detect_markers_v4.py)
- v5 — [`src/preprocessing/detect_markers_v5.py`](../src/preprocessing/detect_markers_v5.py) (synthetic-template NCC)
- v6 — [`src/preprocessing/detect_markers_v6.py`](../src/preprocessing/detect_markers_v6.py) (real-template NCC + fiducial-snap) — **current best**
- template maker — [`src/preprocessing/make_fiducial_template.py`](../src/preprocessing/make_fiducial_template.py) (crops the real fiducial v6 uses)

Each writes overlays to its own folder (`marker_vis/`, `_v2/`…`_v6/`) and a JSON to `logs/`, so all
can be compared on the same images. All are **localization only** (no IDs, no triangulation) and
**read-only** on the dataset. Benchmarks below are on `field_A/20250609` (119 phone images, 6
physical markers).

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

> **After v4 we stopped adding rules.** v1–v4 are all *hand-crafted* cues (white, ellipse, dark
> disk, contrast). Every cue we add has *canopy lookalikes* that pass it and *marker exceptions*
> (occluded/tilted/distant) that fail it — so tuning one knob just trades recall for precision.
> v5–v6 change the *kind* of method: **template matching** (Option A).

---

## v5 — synthetic-template matching (NCC)

**Cue:** instead of *describing* the fiducial with rules, *show* the computer a picture of it (a
**template**) and slide it over the image measuring how well it lines up. Where the image looks
like the fiducial, the match score spikes; canopy doesn't.

**Algorithm:**
1. **Build a synthetic bullseye template** (drawn in code): white plate → grey disk → white dot.
   The coded ID ring is left out on purpose (it *varies* per marker; the fiducial core does not), so
   one template matches all 6.
2. **Multi-scale NCC** — a fiducial is bigger up close, smaller far away, and `cv2.matchTemplate`
   is *not* scale-invariant, so match the template at a bank of disk radii. NCC = **Normalized
   Cross-Correlation** (`TM_CCOEFF_NORMED`): score in [−1, 1], normalized so a correct pattern
   scores high whether the spot is sunny or shaded (lighting-robust by construction).
3. **Matching runs on a downscaled copy** (≈0.35×) for speed (matchTemplate cost ∝ pixels); refine
   on full res. *(This was the fix for an 11-min/5-image first attempt.)*
4. **NMS** → one peak per fiducial; **contrast guard** (disk darker than plate, relative) + a
   **white-plate surround gate** to reject canopy.

**Result — promising but weak separation.** It runs (2 s for 5 images after the downscale fix), and
the white-plate gate cut most canopy. But the **synthetic template is crude**: real fiducials only
reached NCC **~0.73** while canopy reached **~0.6** — a thin ~0.13 margin, so no threshold cleanly
separates them. And the **center refine was wrong**: it snapped to the *brightest* pixels near the
peak, but the plate is the brightest thing, so it dragged the center onto plate-white instead of the
fiducial. Full run: mean 8.3 markers/image, but with many false positives and **multiple drifted
dots per marker** (different scales' peaks landing on different bright plate spots, too far apart for
NMS to merge).

> **Drove v6:** the *idea* (template matching) is right, but a drawn template is too generic and the
> centering is broken.

---

## v6 — real-template matching + fiducial-snap (current best)

**Cue:** same as v5, but match a **real** cropped fiducial (actual pixels) instead of a drawn one,
and fix the centering by locating the *dark disk* (not the bright plate).

**Algorithm:**
1. **Real template:** [`make_fiducial_template.py`](../src/preprocessing/make_fiducial_template.py)
   crops one genuine fiducial (from a v4 detection — v4 centers are reliable) and saves it at a
   canonical size. v6 resizes *that* to each radius in the bank. A patch of real pixels (true grey,
   soft edges, partial ring, sensor noise) correlates **much** more strongly with real fiducials and
   less with canopy — real scores now reach **0.85–0.98** (a wide margin, not 0.73).
2. **Multi-scale NCC at 0.5×** (better far-marker recall than v5's 0.35×), NMS, contrast guard,
   white-plate gate.
3. **Fiducial-snap (fixes the center):** in a local window around each peak, find the round **solid
   dark disk** (the *plate* is bright → excluded by construction; the opposite of v5's bug), require
   it near the peak, and prefer the most disk-like blob (circularity × solidity) so **coded arcs
   aren't grabbed**. A peak with **no real dark disk → dropped** (kills canopy FPs).
4. **Ellipse-fit center:** fit an ellipse to the disk contour and take its center — the surveyed
   point by design, robust to a faint dot or a partly-occluded disk (a bright-pixel centroid drifts).
5. **Post-snap dedup:** NMS ran on the *raw* peaks, so two peaks that later snap to the **same**
   fiducial both survived → duplicate dots. A second pass after snapping merges centers within
   `3.5×radius`, keeping the higher score → duplicates gone.

**Result — best version, but still not good enough.** On `field_A/20250609`: 0 duplicate clusters,
**100% fiducial-snapped centers**, empty frames down to **2/119**, ~95% of *sampled* detections on
real plates, ~2 clean unique markers/image. The real template + fiducial-snap genuinely fixed the
v5 margin and centering problems. **But** on cluttered frames it still fails badly — e.g.
`IMG_20250609_112223.jpg` proposes 5 markers of which **only 1 is correct** (rest canopy FPs, real
markers missed where wheat occludes the plate), and some centers are still slightly off. The
recall↔precision seesaw (relax the white gate to catch occluded plates → more canopy FPs) is
**structural**, not a tuning bug.

> **Drove the decision to stop:** even a real template + every gate + ellipse-fit can't get high
> recall **and** precision **and** a correct center on a cluttered, variable canopy. **→ Option B.**

---

## Side-by-side

| Version | Cue / theory | Center quality | False positives | Recall | Verdict |
|---|---|---|---|---|---|
| **v1** | white square plate | fuzzy (dark centroid) | many | misses tilted | bad |
| **v2** | ellipses (rings) | ⚠️ sometimes on an arc | few | good | finds plates, wrong center |
| **v3** | fiducial disk | ✅ exact | ✅ ~none | ❌ low | precise but can't find them |
| **v4** | v2 find + v3 center | ✅ exact (100%) | ✅ dropped | ⚠️ moderate | best heuristic |
| **v5** | synthetic template (NCC) | ❌ drifts to plate-white | many | good | template idea right, too crude |
| **v6** | real template + fiducial-snap | ✅ ellipse-fit (100%) | ⚠️ low but present | ⚠️ moderate | **best overall, still not enough** |

Benchmarks on `field_A/20250609` (markers/image): v2 ≈ mean 2.1, v3 ≈ mean 0.8 (58 empty
frames), v4 ≈ mean 1.5 (17 empty frames), v5 ≈ mean 8.3 (FPs + duplicate dots), **v6 ≈ mean 2.0
clean unique (2 empty frames, 0 duplicates, 100% fiducial-snapped, ~95% sampled precision)** — yet
still ~1/5 correct on the worst cluttered frames.

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

## Two ceilings — and the decision to go learned (Option B)

This project hit the wall **twice**, and that's the whole story:

**Ceiling 1 — hand-tuned heuristics (v1–v4).** Every cue we add (white, ellipse, dark disk,
contrast, Otsu) has *canopy lookalikes* that pass it and *marker exceptions* (occluded/tilted/
distant) that fail it. Tuning one knob trades recall for precision and back. No hand-crafted rule
set gets high recall **and** precision **and** a correct center at once. So we changed the *kind* of
method → **Option A: template matching**.

**Ceiling 2 — template matching (v5–v6, Option A, now BUILT).** Option A genuinely helped: the
**real** template (v6) widened the NCC margin (0.73 → 0.98), the **fiducial-snap** fixed the center,
**ellipse-fit** stabilized it, and **dedup** removed duplicates — v6 is the best version by far.
**But it still isn't good enough.** On cluttered frames it's ~1/5 correct (`IMG_..._112223.jpg`),
because the underlying recall↔precision conflict is *structural*: a template can score a fiducial
high, but a partly wheat-occluded plate loses the white-surround signal, and loosening the gate to
catch it lets canopy back in. A single rigid template (one appearance, one tilt) can't represent the
full range of marker appearances on a live canopy.

**Why a CNN is the answer (Option B — next).** A template is a *single, fixed* correlation filter. A
CNN is the **learned, deep, nonlinear generalization** of that idea — *many* filters learned from
data, stacked in layers, so it tolerates the lighting/tilt/occlusion/blur variation that breaks a
rigid template. The repo already has a **YOLO** pipeline to repurpose: hand-label the 6 markers
across ~30–40 images (bounded effort), train a marker detector, then run the **v6 fiducial-snap +
ellipse-fit *inside* each predicted box** for the sub-pixel center — the snap is far more reliable
there because the box already guarantees a marker is present, so canopy can't interfere. Cost: the
manual labeling.

**Status:** v1–v6 are kept for the record; **Option B (a trained CNN/YOLO marker detector) is the
next approach.** Remember the downstream safety net (see [`MARKER_INTEGRATION_PLAN.md`](MARKER_INTEGRATION_PLAN.md)
§11c): per-image perfection isn't required — multi-view triangulation rejects FPs (they don't
intersect consistently) and recovers markers missed in some views from others (≥2-views rule). So
"good per-view detections from a CNN" + "multi-view geometry" is the combination expected to close
the gap.

See [`MARKER_DETECTION_STAGE_A.md`](MARKER_DETECTION_STAGE_A.md) for the detailed v1 spec, and
[`MARKER_INTEGRATION_PLAN.md`](MARKER_INTEGRATION_PLAN.md) for how localization feeds the rest of
the marker pipeline (IDs → triangulate → distances → compare to ground truth).
