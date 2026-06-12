# SAHI Explained — Slicing Aided Hyper Inference for Wheat-Head Detection

A from-scratch explanation of how SAHI works, why we'd use it for phone images,
and answers to the practical questions that come up: **how many tiles, what
about heads split at tile edges, what about the same head detected twice, and
how to avoid downscaling fine wheat detail.**

SAHI is a **training-free** wrapper around an existing detector
(our GWC-trained YOLOv5 — see `MASK_GENERATION_OPTIONS.md`). It changes only
*how* the detector is run at inference, not the weights.

Reference: Akyon et al., *Slicing Aided Hyper Inference and Fine-tuning for
Small Object Detection*, arXiv:2202.06934.

---

## 1. The problem it solves

A detector has a **fixed network input size** called `imgsz` (e.g. 640 px).
Whatever image you give it is **resized to `imgsz` before the first layer**.
That resize is the "downscale."

Our phone images are ~3850 × 2928 px with **many small, dense, overlapping
heads**. Running YOLO normally means:

```
3850 px image  ──resize──►  1280 px input      (a 3× downscale)
```

A head that is ~67 × 161 px in the original becomes ~22 × 53 px by the time YOLO
sees it — small, blurred, and the fine awn detail is gone. Small objects are
exactly what gets lost when a big image is squashed down. That's why phone YOLO
looks bad while FIP (overhead, fewer/larger heads, lossless PNG) looks fine.

**SAHI's idea:** instead of squashing the whole image into one pass, cut it into
**tiles** and run the detector on each tile, so each head keeps (most of) its
original pixels.

---

## 2. How the tiling works + how many tiles

SAHI lays a regular grid of **overlapping** tiles. The grid is controlled by
three numbers: **slice size**, **overlap ratio**, and the **image size**. The
step between tile origins is:

```
step = slice_size × (1 − overlap_ratio)
```

Worked example — phone image 3850 × 2928, slice 640, overlap 0.2:

```
step = 640 × 0.8 = 512 px
cols = ceil((3850 − 640) / 512) + 1 = 8
rows = ceil((2928 − 640) / 512) + 1 = 6
→ 8 × 6 = 48 tiles per image
```

So **48 detector passes instead of 1.** Slice size is the main knob:

| slice | overlap | tiles / image |
|---|---|---|
| 640  | 0.2 | 48 |
| 1024 | 0.2 | 20 |
| 1280 | 0.2 | 12 |

Bigger slices → fewer tiles. (Whether that costs detail depends on `imgsz` —
see §3.)

---

## 3. Downscale, `imgsz`, and how to avoid scaling entirely

This is the part that matters for preserving fine wheat detail.

- Each **tile** is itself resized to the detector's `imgsz` before inference.
- **Scaling happens only when `slice_size ≠ imgsz`.**
- **If `imgsz = slice_size`, the tile enters YOLO at 1:1 native pixels — zero
  resampling, no artifacts, fine awns preserved.**

Two things are conserved, which together explain every trade-off:

```
detail (pixels-per-head at inference)  ∝  imgsz / slice_size
total compute                          ∝  num_tiles × imgsz²
```

### Native-resolution options (no scaling — set `imgsz = slice`)

| slice / imgsz | downscale | head long-axis seen by YOLO | tiles | total compute |
|---|---|---|---|---|
| 640 / 640   | none | 161 px | 48 | ~12× one 1280 pass |
| 1024 / 1024 | none | 161 px | 20 | ~12× |
| **1280 / 1280** | **none** | **161 px** | **12** | **~12×** |

All three preserve full detail; the total compute is the **same** (it equals
"process the whole image at native resolution" — a fixed pixel budget). But
**1280/1280 needs only 12 tiles**: fewer seams (good for long heads, §4) and
less per-tile overhead, so it's the *fastest* native option in practice. The
only cost of bigger native tiles is **VRAM per tile** (1280² ≈ 4× the activation
memory of 640²), which is fine on a 16 GB card.

➡ **Recommendation to avoid scaling: use the largest native tile VRAM allows,
with `imgsz = slice_size` (e.g. 1280/1280 → 12 tiles, zero scaling).**

### The compute-saving option (only if you *accept* some downscale)

If runtime ever matters more than maximum detail, make tiles bigger than
`imgsz`:

| slice / imgsz | downscale | head seen | tiles | compute |
|---|---|---|---|---|
| 1280 / 640 | 2× | 80 px | 12 | **¼ of 640/640** |

Still far better than today's 53 px, at a quarter of the native cost. But this
*does* resample, so it's the trade-off to reach for only if needed — not the
default for fine wheat detail.

---

## 4. The edge problem — heads split at a tile boundary

A hard grid would slice a head straddling a seam into two meaningless
fragments. **Overlap is the fix.** With slice 640 and overlap 0.2, neighbouring
tiles share a **128 px band**:

```
 tile A          tile B
┌───────────┬┄┄┐
│           ┊  │      the 128px overlap band  ┊┄┄┊
│        ●  ┊● │      head ● is cut at A's right edge,
│           ┊  │      but sits WHOLE inside B's left edge
└───────────┴┄┄┘
```

A head clipped at tile A's edge falls **fully inside** tile B (B reaches 128 px
back into A's area). So as long as a head is **smaller than the overlap band**,
it appears intact in at least one tile — that's the design guarantee.

⚠ **Wheat-specific caveat:** our heads are ~67 × 161 px (`MASK_SIZE_ANALYSIS.md`).
The 67 px width fits inside a 128 px band, but the **161 px long axis does not**.
A head lying lengthwise across a seam can exceed the overlap. **Fix: raise
overlap to ~0.3–0.4** (192–256 px band) for wheat, or use a bigger slice (fewer
seams). This is a knob we must tune, not leave at default.

---

## 5. The same head detected twice — the merge step

Overlap fixes splitting but creates duplicates: a head in the shared band is
detected in **both** tiles. SAHI resolves this in the **aggregation** step —
every tile box is mapped back to global image coordinates, then de-duplicated:

- **NMS** (Non-Max Suppression): of two overlapping boxes, keep the
  higher-confidence one, drop the other.
- **NMM** (Non-Max **Merging**): *merge* overlapping boxes into their union
  instead of dropping. **Better for seams** — if a head is a full box in one
  tile and a fragment in another, NMM stitches them.
- **Match metric — IOU vs IOS:** IOU (intersection-over-union) underestimates
  overlap when one box is a small fragment; **IOS** (intersection-over-smaller)
  is more robust at seams. Prefer **IOS + NMM** for dense wheat.

Pipeline: **slice (with overlap) → detect per tile → shift boxes to global
coords → IOS/NMM merge.** Overlap prevents the split; the merge removes the
duplicates the overlap creates.

---

## 6. Residual edge case + the safety net

The one case still problematic: a head **larger than the overlap band** landing
on a seam — caught as two fragments, neither complete. Mitigations, in order:

1. **Bigger overlap** (cheapest knob; covers our 161 px heads).
2. **Bigger slice** (fewer seams overall).
3. **`perform_standard_pred=True`** — SAHI optionally *also* runs one normal
   full-image (downscaled) pass and merges it with the sliced results. The full
   pass sees every head uncut, so even a badly-seamed head still gets a complete
   box from it. Costs one extra inference; strong backstop for large heads.

So it's not "tiled vs whole image" — SAHI can do **both and merge**: small-head
recall from the tiles, large-head completeness from the full pass.

---

## 7. Is the overhead a problem for us?

Probably not. This is **offline, one-time mask generation** over a few hundred
phone images, not a realtime system. Our YOLO phase is already GPU-batched and
pipelined, so the cost is ~12× the *compute* of a single pass (with native
tiles), not 12× the wall-clock per image — tiles batch together. A fast YOLO at
~12× over a few hundred images is minutes. Get it working first, measure the
actual minutes, optimize the slice/`imgsz` ratio only if it bothers you.

---

## 8. Integration note (our codebase)

We use a **vendored** YOLOv5 (`src/mask_generation/yolov5/`), not the pip
package, so SAHI needs a **small adapter** to its detection-model API rather than
the one-line `AutoDetectionModel.from_pretrained(model_type="yolov5", …)` path.
The adapter wraps our existing inference call so SAHI can drive it per tile.
Feasible, not free — see `MASK_GENERATION_OPTIONS.md` §2a.

Also: we **can't measure** any of this on phone yet (no phone ground truth →
`metrics_yolo_v1.py` can't score it). SAHI's gain is qualitative until GT exists
(blocked on the supervisor's labeling method).

---

## 9. Quick-start settings (when wiring it)

```
slice_height = slice_width = 1280     # = imgsz → no scaling, native detail
imgsz        = 1280                   # match slice exactly
overlap_ratio = 0.3                   # covers the ~161px elongated heads
postprocess  = NMM, match_metric = IOS, match_threshold ≈ 0.5
perform_standard_pred = True          # full-image safety net for big heads
```

→ ~12 tiles/image on a 3850×2928 phone frame, zero downscaling, seams covered,
duplicates merged. Tune `overlap` up if heads are still split, drop `imgsz`
below `slice` only if you need it faster.
