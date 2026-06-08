# JPEG compression of the phone captures — what it is and how much it hurts

**Script:** [`src/analysis/analyze_jpeg_quality.py`](../src/analysis/analyze_jpeg_quality.py) (read-only).
**Raw numbers:** [`analysis_results/jpeg_quality.json`](analysis_results/jpeg_quality.json).
Re-run with `python src/analysis/analyze_jpeg_quality.py`.

---

## 1. What is JPEG and what does it do to our images?

JPEG is a **lossy** image format: to make files small it **permanently throws away** image
information it judges hard to see. Our phone (Samsung Galaxy S20 FE) saves every frame as JPEG.
The encoding does three things; the third is the irreversible one:

1. **Colour transform + chroma subsampling.** The image is split into brightness (luma) and
   colour (chroma). JPEG then usually **halves the colour resolution in both directions**
   (`4:2:0` subsampling) — i.e. it stores one colour sample per 2×2 pixel block instead of per
   pixel. Brightness is kept full-res; colour is coarsened.
2. **8×8 DCT.** Each channel is cut into **8×8-pixel blocks**, and each block is converted to
   frequencies (a smooth average + progressively finer detail).
3. **Quantization (the lossy step).** The fine-detail (high-frequency) coefficients are **divided
   and rounded**, often to zero. This is what deletes detail. How hard it rounds is set by the
   **quality factor** (0–100): high quality ≈ small divisors ≈ little loss; low quality ≈ large
   divisors ≈ much loss. **This step cannot be reversed** — decoding gives pixels back, but the
   rounded-away detail is gone for good. (Re-saving as PNG only preserves the already-damaged
   pixels; it restores nothing.)

### Why this matters specifically for a wheat field
- A **wheat head is only a few pixels wide**, and an **awn** (the thin bristle/whisker) is a
  near-single-pixel high-frequency structure. High-frequency detail is exactly what quantization
  attacks first → fine wheat structure is the most exposed.
- An **8×8 block can be larger than a whole wheat head**, so a head's edge can get quantized
  together with its background → the true boundary is **smeared or shifted by a few pixels**.
- Wheat heads are often told apart from leaves by **colour**; `4:2:0` halves colour resolution, so
  colour edges between head and leaf get blurred.

### Effect on each pipeline step
| Pipeline step | Sensitivity to JPEG | Why |
|---|---|---|
| **COLMAP SfM** (preprocessing) | low–moderate | SIFT is fairly robust to mild JPEG; the bigger phone problem is repetitive vegetation, not compression. |
| **YOLO detection** | moderate | Boxes can tolerate slightly fuzzy edges, but colour/edge loss on few-pixel heads costs some recall/precision. |
| **SAM masks** | **high** | Masks trace **edges**. Block/ring artifacts + halved colour resolution shift the traced boundary by pixels on objects only a few pixels wide. |
| **3D segmentation** (FlashSplat) | **high (inherited)** | It lifts the 2D SAM masks into 3D — imprecise masks in → imprecise 3D head IDs out. The 2D error **compounds**. |
| **3DGS reconstruction** | low | Training averages ~90 views, so per-image block noise is largely washed out; geometry/PSNR are only mildly affected. |

**Takeaway:** JPEG hurts the **detection/segmentation branch** much more than the 3DGS branch —
because masks live on edges and the heads are tiny.

---

## 2. How we measured it

The JPEG file **stores the evidence of its own compression in its header**, so we don't have to
guess. For each phone session we sampled images and read, without altering anything:
- **Quantization tables** (`img.quantization`) → estimate the **quality factor** and how hard the
  **high-frequency luma** coefficients were quantized (`luma-HF`: higher = more fine detail killed).
- **Chroma subsampling** (`4:2:0` vs `4:4:4`).
- **EXIF** software tag.
- A measured **block-artifact ratio**: mean pixel gradient *on* the 8-px block boundaries vs *off*
  them. `> ~1.15` means visible blocking is actually present.

---

## 3. Results

| session | n imgs | size | subsampling | quality~ | luma-HF | block ratio | software |
|---|--:|--:|:--:|--:|--:|--:|:--|
| phone/field_A/20250609 | 119 | 4032×3024 | **4:2:0** | 91 | 20.1 | 1.06 | — |
| phone/field_A/20250618 | 93 | 4032×3024 | **4:2:0** | 91 | 20.1 | 1.04 | G781B |
| phone/field_D/20250523 | 64 | 4032×3024 | **4:2:0** | 91 | 20.1 | 1.03 | G781B |
| phone/field_D/20250530 | 85 | 4032×3024 | **4:2:0** | 91 | 20.1 | **1.13** | G781B |

(`G781B` = Samsung Galaxy S20 FE 5G, 12 MP.)

### Interpretation

- **Luma compression is moderate, not catastrophic.** Quality ≈ **91** is a typical high phone
  default. The high-frequency luma quantizer averages **20**, vs ~99–120 in the quality-50
  baseline table — so the 8×8 DCT *does* discard some fine detail, but at q91 the loss is **modest**,
  not the detail-annihilation a q75 image would show. Block ratios ~**1.03–1.06** confirm it:
  blocking is barely above noise (below the 1.15 "visible" threshold). So the awn/texture loss from
  luma DCT is real but **smaller than feared**.

- **The genuine JPEG cost here is `4:2:0` chroma subsampling** — present on **every** session.
  Colour resolution is literally halved in both directions. For tiny, colour-distinguished wheat
  heads this is the part that actually blurs the colour edges YOLO/SAM rely on. **This — not the
  luma DCT — is the real mask-precision cost.** So the right way to rank it: JPEG is a *moderate*
  issue for masks, **dominated by chroma subsampling, not by 8×8 luma blocking.**

### Two cross-checks
1. **`field_D/20250530`** has the highest block ratio (1.13) but identical quality (91). It's the
   same session [`analyze_sharpness.py`](../src/preprocessing/analyze_sharpness.py) flagged as **3×
   blurrier**. Blur lowers the off-boundary gradient, which *inflates* the ratio — so this is a
   blur artifact, not extra compression. Two diagnostics agreeing on the same outlier.
2. **Count mismatch:** this session has **85 JPEGs in `input/` but only 63 registered** in COLMAP →
   22 images failed SfM, consistent with the blurry-outlier story (see
   [`COMPARE_TO_AGISOFT_RESULTS.md`](COMPARE_TO_AGISOFT_RESULTS.md)).

---

## 4. What to do about it

- **Existing captures are irreversible.** q91 luma is acceptable; the `4:2:0` colour downgrade is
  baked in. Don't bother re-saving to PNG — it restores nothing.
- **Don't re-compress in the pipeline.** Already handled: `preprocess_uniform_size.py` saves with
  Pillow `quality="keep"` so it doesn't stack a second lossy round on top.
- **Best fix when re-shooting:** capture **DNG/RAW** (OpenCamera supports it) — no lossy
  quantization and no chroma subsampling at all. This is the only way to keep the fine wheat detail.
- **If RAW isn't practical:** force **maximum quality + `4:4:4`** (no chroma subsampling), which
  removes the dominant cost identified here. Also keep HDR off (its frame merging adds artifacts and
  the resolution-mixing we already crop around).

---

## 5. Caveats

- `quality~` is an **estimate** (inverts the standard IJG luma table; assumes IJG-derived tables,
  which most phone encoders use). Good enough to separate "near-lossless ~95" from "destructive
  ~75", not an exact encoder setting.
- The block-artifact ratio is measured on a center crop and is **confounded by image sharpness**
  (blurry images inflate it), as `20250530` shows.
- This analysis is about **compression**, separate from **sharpness** (`analyze_sharpness.py`) and
  **SfM sparseness** ([`SPARSENESS_ANALYSIS.md`](SPARSENESS_ANALYSIS.md)) — three different
  input-quality axes.
