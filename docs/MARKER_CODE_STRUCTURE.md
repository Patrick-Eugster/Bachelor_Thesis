# 12-bit Coded Marker Structure — Codes, Rotations, Hamming, Legal Set

Findings from investigating the Agisoft 12-bit coded targets (CCTs) we decode with the
vendored `src/preprocessing/cctdecode/` library. This is the "coding theory" side of the marker
work — how the IDs are built, why some misread into each other, and how we filter junk. For the
detector pipeline see [`MARKER_DETECTION_CCT.md`](MARKER_DETECTION_CCT.md); for the COLMAP
integration goal see [`MARKER_INTEGRATION_PLAN.md`](MARKER_INTEGRATION_PLAN.md).

---

## 1. What the code is, physically

A coded target is: white square **plate** → solid dark central **disk** (with a tiny **white dot**
at the exact center = the surveyed point) → a ring of dark **arcs** (the **code ring**), concentric
with the disk. The arcs encode the ID.

The decoder samples **N = 12 points** evenly around the code ring (at 2.5× the disk radius). Each
sample is a bit: dark segment = `1`, light gap = `0` (for `color='black'` markers the bits are
swapped after sampling). So one marker = a **12-bit binary ring**.

**Arcs ↔ bits:** consecutive `1` bits merge into one arc; the **number of arcs = number of runs of
consecutive 1s** around the ring; arc width = run length. Example:

```
code  85  ring bits: 1 0 1 0 1 0 1 0 0 0 0 0   → 4 separate single arcs
code 117  ring bits: 1 0 1 0 1 1 1 0 0 0 0 0   → 3 arcs (the last three 1s merge into one wide arc)
```

(Reference renders, drawn with the library's `DrawCCT_black`: `cctdecode/CCT_IMG_12_Black/85.png`
and `117.png`.)

---

## 2. Rotation invariance — `B2I` and "canonicalize"

A marker is a **circle with no fixed "up."** Photograph it rotated and the 12 bits are the *same
cyclic sequence* starting at a different position. To give the same physical marker the **same
number regardless of orientation**, the decoder's `B2I` ("Binary-to-Integer") function tries **all
12 rotations** and returns the **smallest integer** as the official name.

> **Canonicalize** = pick one standard representative for the whole family of rotations of a pattern.
> Here the representative is the rotation with the **minimum integer value** ("minimum over all 12
> rotations").

Example: a single dark segment `100000000000`; rotating moves the `1` around; the rotation that puts
it in the lowest bit gives value `1`, so every rotation of "one dark segment" → canonical code **1**.

Consequence: every code we report (77, 85, 89, …) is already a **rotation-canonical** value. To
compare two codes correctly you must use a **rotation-aware Hamming distance** = the minimum, over
all 12 rotations of one code, of the bit-differences to the other.

---

## 3. The legal code set (necklaces) — and why it is NOT a junk filter

Not all `2^12 = 4096` bit patterns are distinct markers — many are rotations of each other. The set
of **legal codes = the rotation-canonical representatives** (a.k.a. binary necklaces of length 12):

```
12-bit legal codes (necklaces): 352   (out of 4096)
```

**Important correction (verified):** membership in this 352-set does **NOT** separate real markers
from junk. The decoder always returns a *canonical* value (`B2I` = min over rotations), so **every
decoded code is automatically one of the 352** → a necklace-membership test always passes. Worse, the
canopy "all-ones" junk codes (`511 = 000111111111`, `1023`, `1535`, …) are themselves perfectly legal
necklaces. So a "valid-dictionary filter" built on the 352 is a **no-op**.

→ The thing that actually removes junk is the **per-plot manifest** (§5.1 / §6): the specific codes
deployed in this plot. The necklace set + Hamming below are still useful — for the future generator
tool and for flagging near-neighbour misreads — just not as a junk filter.

---

## 4. KEY FINDING — legal ≠ well-separated

Being a legal code does **not** mean codes are far apart in Hamming space:

```
min pairwise (rotation-aware) Hamming over ALL 352 legal codes: 1
number of legal-code pairs at Hamming distance 1:               2004
```

So **many legal codes are a single bit-flip apart.** A single misread segment (glare, a fly, blur
filling one gap) can turn one legal code into **another legal code**. The whitelist alone therefore
cannot fix a 1-bit misread — both the right and wrong codes may be legal.

### 4.1 The 85 ↔ 117 case (target 5)

- Both `85` and `117` are **legal** codes.
- Rotation-aware **Hamming(85, 117) = 1** — they differ by exactly one ring segment (the gap between
  two arcs in 85 fills in → the arcs merge → 117; see the bit rows in §1).
- target 5 reads **85** in the sharp session (`field_A/20250609`, 10 views) and **117** in the harder
  one (`field_A/20250618`, 3 views). → **117 is a single-bit misread of 85.**
- 117 is Hamming-1 from **three** of our real markers at once (85, 101, 113), so **Hamming alone
  cannot uniquely correct it** — only geometry (which 3D marker is at that location) resolves it.

---

## 5. Our deployed markers — GROUND TRUTH from the spec PDF

The reference sheet `reference/agisoft/Coded_12bit_15cm-square_13cm-outer-circle_.pdf` has **6 pages,
one coded target per page, labeled by Agisoft target number**. Decoding each page (clean, frontal →
unambiguous) gives the **authoritative code for every target**:

```
target 1 → 113    target 2 → 105    target 3 → 89
target 4 → 101    target 5 →  85    target 6 → 77
```

So the deployed set (= the **manifest**) is, definitively:

```
{77, 85, 89, 101, 105, 113}
```

This **resolves target 5 = 85** (not 117 — see §4.1) from the source artifact, and confirms the
location-based ID↔target map below. It also validates the decoder: all 6 decode cleanly off the PDF
and match, so field misreads are purely image-quality (glare/blur/tilt), not a decoder bug. A
user/farmer supplying their marker PDF is the realistic way to provide the manifest — we decode it
automatically (6 pages → 6 codes).

- **All 6 are legal** codes (each equals its own rotation-canonical form). ✓
- **Mutually Hamming ≥ 2** (min pairwise distance = 2).

**Why min-distance 2 matters (good news):** a single-bit misread of one real marker can **never**
equal another real marker (that needs ≥2 flips). It always lands on an **outsider** code (like 117,
which is not deployed). So with our manifest we can always **detect** a misread (it falls outside the
set) — we just can't always **auto-correct** it uniquely (needs ≥3 separation for that; see §7).

Marker ↔ Agisoft target map (location-based, established on the clean 20250609 session):

```
113 ↔ target 1   105 ↔ target 2   89 ↔ target 3
101 ↔ target 4    85 ↔ target 5   77 ↔ target 6
```

Note: **our code value ≠ Agisoft's "target N" label** — Agisoft's labels are arbitrary project names,
not the 12-bit code. The map above is a per-session location correspondence, not equality.

---

## 6. How we use this — the filtering / ID-cleanup stack

In order, before/around triangulation:

1. **Plot manifest filter (the real junk filter):** keep only the codes actually deployed in this
   plot (`{77,85,89,101,105,113}`, decoded from the spec PDF). Drops both wheat-junk (`511, 1023, …`)
   AND near-neighbour misreads (`117`, `1535`) in one step. **Implemented as v8's `id_filter=manifest`
   mode** (the default), replacing the fragile `keep_top_k` view-count heuristic. Dropped detections
   are kept (with locations) in the JSON's `per_image_dropped`. NOTE: a necklace "dictionary" filter
   does **not** work here (§3) — the manifest is what does the job.
2. **Triangulation by location = the final arbiter:** group detections by 3D location. The manifest
   *drops* a misread like 117; triangulation can instead *recover* it by **snapping 117→85** (it sits
   at target 5's 3D location) — resolving systematic misreads that neither majority vote **nor**
   Hamming alone can fix (117 is Hamming-1 from 85, 101 *and* 113, so only geometry disambiguates).

   **The snap/seed is PURELY geometric — the Hamming guard was dropped (`snap_hamming_max=0`).** A
   dropped detection is reassigned to a marker if it lands on that marker's 3D reprojection (within
   tol) and survives RANSAC, *regardless of its decoded code*. Reason: under heavy occlusion a misread
   can flip more than one bit, so a Hamming≤1 guard would wrongly *exclude* a real recovery; and once
   we have the 3D point, the wrong code carries no useful information. RANSAC + the reprojection
   threshold are the real safeguards. (Set `snap_hamming_max>0` to re-enable the code guard.) See
   `triangulate_markers.py` (snap = attach to a *solved* marker; seed = rebuild an *unsolved* marker
   from its leftover misreads, ruling out those that land on already-solved markers).

**Do not rely on majority vote alone:** a misread can be *systematic* (117 read 3/3 at target 5 in
20250618), so the majority within a session can be a wrong/non-deployed code. Membership in the
manifest + geometry must back it up.

---

## 7. TODO (future) — coded-target generator tool

For **future** captures (not our current data — those markers are already placed and can't be
re-done): a tool that, given the 352 legal codes, returns **K codes with the largest mutual Hamming
distance** (greedy farthest-point / max-min selection).

Why it matters: if a deployed set has **min pairwise Hamming ≥ 3**, then every single-bit misread has
a **unique** nearest deployed code → error-correction becomes provably safe and the systematic-misread
ambiguity (§4.1) disappears without needing geometry. Our current markers are only min-distance 2
(enough to *detect* misreads, not always *correct* them), which is why target 5 needed triangulation.

Sketch: enumerate necklaces → build rotation-aware Hamming matrix → greedy max-min subset of size K →
hand back the codes + their `DrawCCT` images for printing. Purely forward-looking; no impact on the
existing pipeline.

---

## 8. Quick reference (numbers)

| thing | value |
|---|---|
| ring bits N | 12 |
| total patterns | 4096 |
| **legal codes** (canonical necklaces) | **352** |
| min Hamming over all legal codes | **1** (2004 pairs at distance 1) |
| our deployed markers | `{77, 85, 89, 101, 105, 113}` |
| min Hamming among our 6 | **2** |
| 85 vs 117 | both legal, Hamming **1** (117 = misread of 85 = target 5) |
| code value vs Agisoft label | **not** equal (label is arbitrary) |
