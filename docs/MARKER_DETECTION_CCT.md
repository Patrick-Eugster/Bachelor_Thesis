# Marker detection — Option C: CCTDecode (coded-target detect-and-decode)

> **Why this exists.** Our hand-rolled localizers (v1–v6, see `MARKER_DETECTION_VERSIONS.md`) all try to
> *recognize* the marker by appearance — shape → ellipse → template → texture. They hit a ceiling: false
> positives on wheat, off-center hits, and misses when the central fiducial is occluded. The decisive
> property we were ignoring is that these are **coded** targets: each marker carries a self-validating
> **12-bit number**. The moment you *decode* instead of just *recognize*, the false-positive problem and the
> ID problem both collapse — a wheat blob cannot produce a valid 12-bit codeword, and a valid decode hands
> us the marker ID for free (that was going to be a whole separate "Stage B"). Reference implementation we
> are adopting: **[poxiao2/CCTDecode](https://github.com/poxiao2/CCTDecode)** (pure Python + OpenCV, MIT-ish,
> no torch, no compilation).

---

## How CCTDecode actually works

The whole thing is one function, `CCT_extract(img, N, R, color)` in `CCTDecodeRelease.py`.
`N` = number of bits (12 for us), `R` = a circularity threshold (the `--threshold` CLI arg, default 0.85),
`color` = `white`/`black` (are the code marks light-on-dark or dark-on-light). Per image it does:

1. **Grayscale → global Otsu threshold** over the *whole* image → a binary (black/white) image.
2. **`cv2.findContours`** on the binary, keep every contour.
3. **Circularity filter**: `R0 = 2·√(π·area)/perimeter`; drop contours with `R0 < R`, or with fewer than 20
   points. → candidate round blobs.
4. **`cv2.fitEllipse`** on each candidate → center, axes, rotation angle. It treats this ellipse as the
   marker's **central dot**, and assumes the coded ring sits at **2.5×** that dot radius (outer edge at 3×).
5. **Affine-rectify**: it warps the elliptical ROI back into a frontal circle using a 5-point affine
   transform (the 4 bounding-box corners + the center). **This is the step v1–v6 never had** — it undoes the
   perspective tilt *before* reading the code, so an obliquely-viewed marker is straightened first.
6. **`CCT_or_not`** (the validation gate): on the rectified circle it samples three radii — inside the dot
   (0.5×) must be uniform, the gap ring (1.5×) must be the opposite value, the code band (2.5×) must contain
   some bits. Junk that isn't actually a CCT fails this and is dropped.
7. **`CCT_Decode`**: samples `N = 12` points evenly around the 2.5× code ring, repeats that over 30
   sub-degree rotations and averages (noise robustness), thresholds at 0.5, then normalizes for rotation via
   a "minimum code" trick → a single **integer ID**.
8. **Output**: `CodeTable = [[code, cx, cy], …]` — i.e. **marker ID + center in original-image pixels** —
   plus an annotated overlay image with the fitted ellipses and the decoded number drawn on.

So out of the box it gives us exactly the two things we want per marker — **a self-validated ID and a
sub-pixel center** — and it explicitly handles tilt. Dependencies are tiny: `opencv-python`, `numpy`,
`pillow`, `matplotlib`, `progress`.

---

## The honest catch — what fights our data

CCTDecode was built for **close-up / drone shots where the CCT is large, dominant, and high-contrast**. Our
markers are ~30–40 px fiducials buried in a busy 4000 px wheat frame. Two specific mismatches:

1. **Global Otsu over the whole frame (step 1).** Otsu picks **one** brightness cutoff for the entire photo.
   A field frame has bright sky, sunlit canopy, shadows, *and* a small bright marker — no single cutoff is
   right everywhere. The cutoff ends up chosen to separate sky-vs-ground (the dominant histogram peaks), not
   marker-vs-background, so the small marker is never cleanly isolated → the detector finds nothing or a sea
   of garbage blobs. **Fix:** replace the front-end with **adaptive/local thresholding** (a different cutoff
   per neighborhood), and/or feed the decoder **crops** around candidate locations instead of full frames.
2. **Radius-ratio assumption (2.5× ring).** CCTDecode hard-codes *its own* geometry. Agisoft's "12-bit,
   15 cm square / 13 cm outer circle" target may place the code ring at a **different ratio** and number the
   bits differently. Consequence: our decoded IDs will be **internally consistent across our views** (all we
   need for triangulation), but may **not equal Agisoft's ID numbers** unless we recalibrate the sampling
   radius / bit order to the Agisoft spec sheet (`reference/agisoft/Coded_12bit_…pdf`). We have decided this
   does **not** matter — relative consistency is enough; a one-time hand map to Agisoft IDs is only needed if
   we later want to look up Agisoft's ground-truth 3D position *by* ID.

Occlusion of the central dot (the v6 failure) still hurts steps 3–4, but the affine-rectify + 30-rotation
averaging tolerate partial damage far better than v6's single-dot snap.

---

## What we keep vs. what we change (customization scope)

The valuable, *correct* part of CCTDecode is the **decode core** — steps 4–8 (ellipse fit → affine-rectify
→ `CCT_or_not` validate → ring-sample → bit-decode). **We keep that unchanged.** What we replace or add:

| Part | Original | Ours |
|---|---|---|
| **Front-end** (find candidate regions) | global Otsu + `findContours` on the full frame | adaptive threshold and/or **crop-seeding** (from v6 detections or a YOLO proposer) |
| **Decode core** (steps 4–8) | — | **unchanged** (this is the whole point) |
| **Wrapper / I/O** | argparse CLI, writes annotated PNG only | Hydra config (`dataset=phone`, `field`, `plot`), `marker_vis_v7/` overlays + `logs/marker_detections_v7.json` matching the v1–v6 output shape |
| **Geometry constants** | its own 2.5× ring ratio | optionally re-measured to the Agisoft spec (only if we want Agisoft-matching IDs) |

This is normal when applying a tool to a harder domain than it was built for: the **algorithm** is reused,
only the **region-proposal front-end** is swapped. Cropping a few real markers by hand (see "Phase 0" in the
plan) is a *diagnostic to validate the decode core in isolation* — not part of the production path.

### Otsu vs. adaptive thresholding (the front-end choice)

- **Otsu = global.** One threshold for the whole image, chosen automatically from the global histogram. Fast,
  great when lighting is uniform and foreground/background are clearly bimodal. *Analogy: one flat water line
  drawn across the entire landscape.*
- **Adaptive (`cv2.adaptiveThreshold`) = local.** For each pixel, the threshold is computed from the mean /
  Gaussian of its small neighborhood (a `blockSize`), so a pixel is "white" if it's brighter than its *local*
  surroundings. Handles uneven illumination, shadows, gradients; slower, has params. *Analogy: the water line
  follows the local terrain.* For a marker that's locally bright relative to nearby canopy but not the
  globally-brightest thing in frame, adaptive isolates it where global Otsu (parked at sky brightness) misses
  it.

---

## A useful prior we have: the marker constellation

The 6 markers sit at **fixed positions on the ground** — they form a rigid 3D constellation that never
changes between views. This is a strong *secondary* check we can layer on top of the decode:

- **False-positive rejection:** a detection whose position doesn't fit the known arrangement is suspect.
- **Occlusion recovery:** if 5 of 6 decode confidently and their layout matches the known constellation, the
  6th is pinned by elimination + expected position even if its code is half-occluded.
- **Cross-view consistency:** once a couple of markers are triangulated, the rigid constellation can be
  matched into every image via camera pose (the "known rigid object" trick) — this folds straight into the
  per-marker **≥2-views** safety net in `MARKER_INTEGRATION_PLAN.md`.

Caveat: the constellation prior is a *downstream* aid (it needs at least a rough pose or several confident
decodes first), not the primary detector. With only 6 markers carrying unique 12-bit codes, the decode is
already strongly unique on its own; the constellation's main value is **recovering occluded markers and
killing false positives**, not disambiguating among the 6.

---

## Status

Decided next approach (supersedes the "go learned / Option B" note at the end of
`MARKER_DETECTION_VERSIONS.md`). Plan is staged:

- **Phase 0 — DONE ✅** (`src/preprocessing/test_cct_phase0.py`, vendored core in
  `src/preprocessing/cctdecode/`). Three isolated checks:
  - **A1** DrawCCT synthetic marker → decoded the expected id (`141`). Core runs in our env.
  - **A2** Agisoft spec-PDF marker #1 → decoded cleanly, the fitted ellipses + code-sampling ring land
    exactly on the marker. **This proves CCTDecode's algorithm is geometry-compatible with the Agisoft
    12-bit targets** — the single biggest unknown, now answered *yes*. (Our decoded number ≠ Agisoft's
    printed label, which is fine — we only need relative consistency.)
  - **B** real field marker hand-cropped from `IMG_20250609_112223.jpg` → **partial**: the decode core
    fired, but on the cluttered/tilted real marker CCTDecode's own contour step latched onto the **code-arc
    segments instead of the true central disk**, and emitted degenerate codes (`4095`, `2047` =
    nearly-all-bits-set). So the **decode core works; the real-image candidate-selection (front-end) is the
    weak link** — exactly as predicted.
  - **Two numpy-2.x / Python-3.12 port bugs fixed in the vendored copy** (see `ATTRIBUTION.md`): the
    `from numpy import *` star-import shadowed builtin `max/min/round/sum` → `TypeError: 'float' object
    cannot be interpreted as an integer`. Removed it (the module already uses `np.`/`math.` prefixes).
- **Phase 1 — NEXT: a v6 × CCTDecode hybrid.** The two methods cover each other: **v6/fiducial-snap finds the
  true center + radius** (v6's strength; its weakness was false positives + no IDs), then run **only
  CCTDecode's rectify → ring-sample → decode around that known center** (CCTDecode's strength: IDs +
  self-validation; its weakness here was *finding* the center). Plus a **valid-ID whitelist (only the 6 real
  markers)** + **degenerate-code filter** (`0`, `4095`, `2047`). Wrapper `detect_markers_v7_cct.py`, v6-style
  overlays + `logs/marker_detections_v7.json`.
- **Phase 2** *(optional, deprioritized)* — recalibrate ring radius / bit order to match Agisoft IDs. Not
  needed: relative ID consistency is sufficient.
- **Phase 3** — emit per-image `(marker_id → pixel xy)` + 3D positions as **GCPs** into COLMAP's native GCP
  bundle adjustment (see `MARKER_INTEGRATION_PLAN.md`, Option D). Validate against the Agisoft reference
  marker projections (received for the phone data).

YOLO/CNN (the old Option B) is kept **in reserve** purely as a crop-proposer feeding the decode core, only if
the v6-seeded front-end can't reach the markers under occlusion.
