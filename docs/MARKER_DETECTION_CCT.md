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

## Terminology (one marker)

- **plate** — the white square the target is printed on.
- **disk** — the solid dark filled circle in the **middle** (an ellipse under perspective). The thing we
  localise.
- **white dot** — the tiny bright dot at the disk's **exact center** = the precise surveyed point.
- **arcs** — the dark curved segments **around** the disk, on the **code ring** (≥2 per marker, encode the
  ID). The code ring is **concentric with the disk** (same center). [from the spec PDF]

---

## Files (all on branch `markers`, all READ-ONLY w.r.t. the dataset)

| File | Role |
|---|---|
| `src/preprocessing/cctdecode/` | vendored [poxiao2/CCTDecode](https://github.com/poxiao2/CCTDecode) (+`ATTRIBUTION.md`); 2 numpy-2.x/py3.12 bugs fixed (removed `from numpy import *` which shadowed builtin `max/min/round/sum`; lazy `progress` import) |
| `src/preprocessing/cct_forced_decode.py` | the decode engine: `find_disk_at` (fill-ratio disk finder), `find_center_concentric` (v8), `decode_at_center(...,finder=)` (forces decode onto a given center, no blob search) |
| `src/preprocessing/detect_markers_v7_cct.py` + yaml | **v7 detector**: v6 proposes centers → `decode_at_center`(find_disk_at) |
| `src/preprocessing/detect_markers_v8_cct.py` + yaml | **v8 detector**: same, but `finder=find_center_concentric` (handles v6-on-arc + occluded disk) |
| `src/preprocessing/overlay_agisoft_markers.py` | draw Agisoft GT projections on the images → `marker_vis_agisoft_gt/` |
| `src/preprocessing/compare_v7_vs_agisoft.py` | score any version vs Agisoft GT: recall + per-target ID map + misses CSVs (`--version v7|v8`) |
| `src/preprocessing/debug_cct_window.py` | per-candidate debug crops: search window + every blob's fill ratio + chosen disk + decoded code |
| `test_cct_phase0.py` / `test_cct_phase0b_gtcrops.py` / `test_cct_phase1_forced.py` | the staged validation tests below |

---

## What we learned, in order

**Phase 0 (`test_cct_phase0.py`) — the approach is viable.** A1 synthetic marker decoded; **A2 the Agisoft
spec-PDF marker decoded cleanly** (proves CCTDecode's algorithm is geometry-compatible with Agisoft 12-bit
targets — the big unknown, answered yes); B a hand-cropped real marker → stock `CCT_extract` latched onto a
**code arc** and emitted junk. ⇒ decode core works, its **own blob-search is the weak link**.

**Phone GT arrived** — `demoanlage2025_v0_additions/<field>/<date>/processed/marker_projections.csv`
(both fields × 19 sessions). Per-image 2D marker positions, **in our undistorted `images/` space**,
sub-pixel. **Asymmetric**: Agisoft has ~zero false positives but **misses many visible plates** — so a
detection we make that Agisoft lacks is **not wrong** (candidate extra); only a GT marker we *miss* is a
problem. ⇒ score **recall + localization** rigorously, treat "extras" as candidates (confirmed by the decode).

**Phase 0b (`test_cct_phase0b_gtcrops.py`) — stock decode at GT centers.** Decoding at the *correct* centers,
stock `CCT_extract` was reliable on **only target 1** (→113); targets 2/3/6 collided on the **arc artifact
"7"**. ⇒ the blob-search reads arcs; we must **force the decode onto the true disk**.

**Phase 1 (`test_cct_phase1_forced.py`) — forced-center decode SOLVES it.** Forcing the decode onto the disk
at the GT center: **all 6 markers decode to distinct, consistent IDs** (t1=113, t2=105, t3=89, t4=101, t5=85,
t6=77), zero collisions. The "7" was purely blob-search grabbing arcs.

**v7 (`detect_markers_v7_cct.py`) — the real detector.** v6 proposes candidate centers → `decode_at_center`
finds the disk there and decodes. The decode self-validates (junk → no valid code → dropped), so v6's false
positives vanish for free. Two fixes made it solid:
- **Disk-vs-arc by FILL RATIO** = blob area ÷ fitted-ellipse area. A solid **disk ≈ 1.0**; a **code arc
  ≤ 0.91**; canopy ≤ 0.65. This is a **shape ratio → SIZE-INDEPENDENT** (no hardcoded pixel distance, which
  was rightly rejected — same lesson as v2's cluster-distance bug). The search window is size-relative
  (× fiducial radius).
- **Report the decoded disk's ellipse center** (not v6's proposal) → **0.7 px median, 97% within 3 px** of GT.
- **Bypass CCTDecode's internal `CCT_or_not`** (`decode_require_valid_cct=false`) — with the fill gate
  guarding precision, that check was redundant *and* rejected tilted/edge markers, costing recall.
- **Result (field_A/20250609):** all 6 markers as the top IDs by view-count; **73% recall of Agisoft**
  (93/127); centers pinpoint; tiny noise tail.

**v8 (`detect_markers_v8_cct.py`) — concentric-consensus + re-centering.** Rejecting an arc threw away the
whole marker; but disk+arcs are **concentric**, so `find_center_concentric` recovers the center from whatever
survives: a solid disk if present (= v7), else **fit the ring to the ≥2 arcs and derive the disk** (works when
the disk is occluded). Plus **re-centering**: if the estimated center is far from where we searched (v6 on an
arc → disk clipped at the window edge), re-search centered on the estimate so the disk is captured cleanly.
- **Result:** **76% recall (97/127)**, target 1 23→27 hits — **the arc-frames (112220–112223) now decode 113
  on the disk** (the reported "id flips 113↔7" bug, fixed). Cost: more extras (65 vs v7's 33) — arc
  reconstruction fires junk IDs at a few non-marker spots (1–5 views; removable by a min-views filter or
  triangulation voting).

**Diagnosis of the remaining ~24 misses** (decode-at-GT-center test): two kinds — (a) decode *fine* at the
true center but the detector missed it because v6 was offset and the window clipped the disk (**fixed by v8's
re-center**); (b) decode **fails even at the perfect center** (target 2/4 in far/small/tilted frames) —
**decode-resolution limited, not center-findable**. Per-marker we are complete regardless (each marker hit in
10–27 views, ≫ the ≥2 triangulation needs).

**ID map (consistent across views, our-id ↔ Agisoft-target):** 113↔T1, 105↔T2, 89↔T3, 101↔T4, 85↔T5, 77↔T6.

---

## Status & next

- **v7 = higher precision** (fewer extras); **v8 = higher recall + fixes the arc/occlusion bug**. Both coexist.
- **Decided "consistent-only" IDs** — we do **not** match Agisoft's numbers; a one-time 6-row hand map covers
  GT lookup if ever needed. So Phase-2 ring-ratio calibration is **dropped**.
- **Next options:** (a) add a **min-views filter** to v8 to drop the noise-tail extras → strict win over v7;
  (b) **triangulation** — fold the per-view detections into 6 3D markers, majority-vote IDs (removes the tail
  *and* the per-view misses), then emit `(marker_id → pixel xy)` + 3D as **GCPs** into COLMAP's native GCP
  bundle adjustment (`MARKER_INTEGRATION_PLAN.md`, Option D).
- YOLO/CNN (old Option B) stays **in reserve** — only if a better proposer is needed; not required so far.
- **PERF — both passes parallelised (`num_workers`, default 8).** v8's per-image work is independent, so
  **pass 1** (decode) and **pass 2** (draw + write overlay PNGs) each run across N worker **processes**
  (`multiprocessing.Pool`, fork; shared inputs handed once via the initializer + `cv2.setNumThreads(1)` so the
  processes don't oversubscribe cores; pass 1 uses ordered `imap` for sensible progress prints, pass 2 uses
  `imap_unordered` since it only tallies counts). `num_workers=1` = old serial path. **field_A/20250609:
  128 s → 37 s (pass 1 only) → 22 s (both passes) = 5.8×, detections byte-identical, all 119 PNGs written.**
  Peak RAM +1.2 GB (≈150 MB/worker — trivial on 35 GB). The remaining ~22 s is the irreducible serial floor
  (template-bank build, worker startup/pickle, JSON write). Default 8 = the Ryzen 7700X3D's physical cores
  (the 16 SMT threads add little on this FP-heavy work).
