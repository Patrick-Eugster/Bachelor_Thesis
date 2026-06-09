# Wheat-head mask sizes & JPEG edge-impact — FIP vs phone

**Goal:** measure how big the SAM wheat-head masks actually are (in pixels), and from that
quantify how much the JPEG compression of the input images can disturb those masks. This tells us
whether JPEG is a real threat to segmentation precision or a minor effect.

**Script:** [`src/analysis/analyze_mask_sizes.py`](../src/analysis/analyze_mask_sizes.py) (read-only).
**Raw numbers:** [`analysis_results/mask_sizes.json`](analysis_results/mask_sizes.json).
Re-run with `python src/analysis/analyze_mask_sizes.py`.
Companion: [`JPEG_QUALITY_ANALYSIS.md`](JPEG_QUALITY_ANALYSIS.md) (what JPEG / 4:2:0 / 8×8 DCT do).

---

## 1. How the masks are stored (so the numbers are not misread)

Each SAM mask is a **full-resolution PNG the same size as the input image** (≈12 MP), almost
entirely **black**, with **one small white blob = one wheat head's silhouette**. In a typical phone
mask only ~**0.06%** of the PNG is white. So when we report a mask "size", we count **only the white
pixels** — the black background is ignored and does **not** inflate anything. The white pixel count
is the head's true silhouette area.

We report, per head:
- **area** = number of white pixels (the real silhouette area).
- **width × length** = shorter × longer side of the tight bounding box around the white blob.
- **perimeter** = number of white pixels that touch the black background (the outline length).
- **fill ratio** = area ÷ bbox area (how much of the bounding box is actually head; the rest is
  background corners, because heads are irregular/elongated, not rectangles).

---

## 2. Results

| dataset | masks | area (px) | W × L (px) | perimeter | fill | JPEG @1px | JPEG @2px |
|---|--:|--:|--:|--:|--:|--:|--:|
| **FIP avg** (7 plots) | ~8,300 | **~6,440** | **~81 × 125** | ~298 | 0.64 | **~5.1 %** | **~10.2 %** |
| phone/field_A/20250609 | 52,594 | 6,786 | 68 × 174 | 385 | 0.60 | 6.2 % | 12.3 % |
| phone/field_A/20250618 | 46,722 | 6,383 | 67 × 158 | 357 | 0.62 | 6.1 % | 12.2 % |
| phone/field_D/20250530 | 6,811 | 5,766 | 67 × 151 | 341 | 0.59 | 6.4 % | 12.9 % |
| **phone avg** (3 good sessions) | — | **~6,310** | **~67 × 161** | ~361 | 0.60 | **~6.2 %** | **~12.5 %** |
| phone/field_D/20250523 (early, outlier) | 968 | 2,389 | 48 × 75 | 182 | 0.68 | 8.3 % | 16.5 % |

**Size takeaways:**
- Heads are **not "a few pixels"** — FIP and phone heads have nearly the **same area** (~6,300–6,400
  px). The earlier worry that heads were tiny is not supported by the data.
- **Phone heads are more elongated** (~67 × 161 px) than FIP (~81 × 125 px): same area, but narrower
  and longer → **longer perimeter** (~361 vs ~298 px) → slightly more edge exposed to compression.
- **fill ≈ 0.60–0.64**: the silhouette fills ~60 % of its bounding box; the bbox alone overstates
  head width, so **`area` is the reliable size metric**, not the bbox.
- The early-season session `20250523` is a clear outlier (tiny 48 × 75 px heads) — heads had barely
  emerged. It is weak on every axis and should be excluded.

---

## 3. What "JPEG @Npx" means — the edge-impact metric (worked example)

JPEG only damages the **edge** of a head — 4:2:0 chroma subsampling and 8×8 DCT ringing blur/shift
the boundary by a pixel or two (see [`JPEG_QUALITY_ANALYSIS.md`](JPEG_QUALITY_ANALYSIS.md)). It
**cannot** touch a pixel buried deep inside a head. So the only pixels a JPEG edge-wobble can get
wrong are the ones **on the outline of the blob**.

Zoom into one white blob, pixel by pixel (`X` = white head pixel, `.` = black background):

```
. . X X . .
. X X X X .
. X X X X .
. X X X X .
X X X X X X
. X X X X .
. X X X X .
. . X X . .
```

Now split the white pixels into two kinds:
- `o` = an **edge** pixel — a white pixel that **touches** the black background (it is on the outline)
- `#` = an **inside** pixel — a white pixel **completely surrounded** by other white pixels

```
. . o o . .
. o o o o .
. o # # o .
. o # # o .
o # # # # o
. o # # o .
. o o o o .
. . o o . .
```

- The `o` pixels form the **outline ring** of the head.
- The `#` pixels are **deep inside**.

**JPEG can wobble the `o` ring; it can never touch the `#` interior.** So the share of the head that
is "at risk" from a 1-pixel edge blur is:

> **JPEG @1px = (number of `o` outline pixels) ÷ (total white pixels, `o` + `#`)**
> = perimeter ÷ area.

For a real head with **area ≈ 6,000 px** and an **outline ≈ 360 px**:

```
360 outline pixels ÷ 6,000 total white pixels ≈ 6 %
```

So **~6 % of the head's pixels sit on the outline ring; the other ~94 % are safe interior.** That 6 %
is `JPEG @1px`. For a 2-pixel-thick ring (chroma + DCT, conservative) it doubles to ~12 % = `JPEG @2px`.

**The intuition:** a **big** blob is mostly `#` interior with a thin `o` outline → small % at risk;
a **tiny** blob is almost all `o` → large % at risk. Our heads are big, so only ~6 % is on the
risky outline — which is why JPEG hurts the masks only modestly. (The tiny early-season heads reach
~8 %, exactly because more of them is outline.)

---

## 4. Caveats — read before citing

**(a) This is an upper bound, not the actual error.** `JPEG @Npx` is the size of the *maximally
affected* outline ring. JPEG does **not** flip every outline pixel by the full 1–2 px, and over-/
under-shoots partly cancel, so the **realized** mask error (e.g. IoU loss) is a **fraction** of this
number. State it as "at most ~6 %", not "6 % error".

**(b) Small per-mask error × thousands of masks can still cause head/mask confusion.** This is the
important practical remark: even though each mask is only ~6 % edge-exposed *on average*, there are
**thousands of heads per plot** (e.g. 8,302 FIP; 52,594 in phone/field_A/20250609) and they are
**densely packed and often touching/overlapping** in the canopy. A few-pixel boundary shift is most
dangerous exactly where two heads are adjacent: it can **merge two heads into one mask, split one
head into two, or shift a mask enough to flip which head it matches** in another view during the
3D segmentation (`run_3d_seg.py` matches masks across views by IoU near a threshold). With so many
heads, even a **low per-head confusion rate yields a non-trivial absolute number** of mis-assigned
or merged heads. So the small average percentage does **not** mean JPEG is harmless for the *count*
and *identity* of heads — it is a real, if subtle, risk concentrated in the dense/overlapping
regions, and worth fixing on the next capture (shoot **4:4:4** or **DNG/RAW**).



---

*Related: [`JPEG_QUALITY_ANALYSIS.md`](JPEG_QUALITY_ANALYSIS.md) (JPEG mechanisms + measured
compression of the phone captures), [`SPARSENESS_ANALYSIS.md`](SPARSENESS_ANALYSIS.md) (SfM input
quality).*
