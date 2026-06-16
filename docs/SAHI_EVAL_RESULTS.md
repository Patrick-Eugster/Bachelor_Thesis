# SAHI Evaluation Results — the floor bug, the single-threshold fix, and the IOS/IOU/CONF merge study

**Headline:** SAHI's first FIP numbers looked *worse* than plain YOLO. The cause was a **confidence-floor bug specific to SAHI's merge**, not SAHI itself. After the fix (collapsing SAHI to a **single confidence threshold**), SAHI **beats plain YOLO on recall** (0.942 vs 0.906) and nearly closes the F1 gap. A follow-up study of the merge metric (IOS vs IOU vs a confidence-aware merge) showed **IOS is the best of the three on FIP**; the remaining weakness — a small head nested in a big diagonal head's box — is **not fixable at the box level** and is parked for the phone data (where side-view diagonal heads make it more common).

Tools used: `eval_compare_3way.py` (SAHI vs YOLO vs GT), `eval_compare_nogt.py`, `sahi_merge_debug.py`. Eval set = the 7 FIP GT-labeled images (one per plot, 1386 GT heads). Matching IoU = 0.35.

---

## 1. The floor bug (why SAHI first looked worse)

The first comparison (`metrics_v2` / `cmp_v2`) gave SAHI **F1 0.825 vs YOLO 0.881** — worse on both precision and recall, while emitting *more* boxes. Diagnosis:

- SAHI's metrics run inherited `conf_threshold_nms_floor = 0.01` from the eval config (a low floor is needed for an AP/PR curve).
- SAHI's merge is **NMM**, which **unions** overlapping boxes and takes the **max** score. The 0.35 good-box filter is applied *after* the merge.
- So at floor 0.01, a swarm of 0.01–0.35 **junk boxes entered the merge** and got unioned into real heads, **reshaping** them into bigger/offset boxes that no longer matched GT.

**This hurts SAHI but not YOLO.** Plain YOLO uses **NMS** (suppress, keep the max box *unchanged*) and applies the floor as the *last* step, so its ≥0.35 boxes are byte-identical regardless of the floor. Verified empirically: YOLO good boxes for the labeled image were **237 vs 237, 0 mismatches** at floors 0.2 vs 0.01.

**Proof on real data:** two SAHI runs identical except the floor — `test461_v2` (floor 0.2) vs `metrics_v2` (floor 0.01) — produced the **same box count (256)** but **32 different boxes**: tight boxes at 0.2 were replaced by **bigger, ~50 px-offset** boxes at 0.01 (the union signature, e.g. 152×161 → 315×345). A clean 0.92-conf head present at floor 0.2 was *gone* at floor 0.01.

---

## 2. The fix — SAHI has ONE confidence threshold

Since SAHI doesn't need an AP/PR curve, the three confidence knobs were collapsed to one:

| before (3 knobs) | after |
|---|---|
| `conf_threshold_nms_floor` (inference floor, low for curve) | **removed** |
| `sahi_merge_conf_floor` (an interim pre-merge drop, then superseded) | **removed** |
| `conf_threshold_good_box` (keep line, 0.35) | **the only SAHI conf** |

**Mechanism:** tiles are run *at* `conf_threshold_good_box` (`model.conf = 0.35`). Every box entering the NMM merge is already ≥0.35, so **nothing sub-threshold can leak into the union and reshape a good box** — the bug is gone *by construction*, not patched. This is also *more correct*: the sub-0.35 boxes were mostly **cut slivers at tile seams**, and dropping them gives tighter, truer boxes.

**Code:** `src/mask_generation/sahi_yolo_sam/sahi_yolo_pipelined.py` (`model.conf` + `merge_preds`), `sahi_merge_debug.py` (`load_yolo`), `configs/mask_generation/method/sahi_yolo_sam.yaml`. **Side effect:** SAHI can no longer be run via `--config-name eval_run` (it set the now-removed `conf_threshold_nms_floor`); regenerate SAHI boxes with a normal run + `only_labeled_images=true` instead. `yolo_sam_v1` is untouched (it genuinely needs the low floor for its AP curve, and NMS can't be corrupted by it).

### Result: v2 (broken) → v3 (fixed)

Same box count, but ~60 boxes moved from misaligned-FP to aligned-TP (the floor bug only *reshaped* boxes, never added/removed them):

| metric | v2 (floor 0.01) | v3 (single conf) | Δ |
|---|---|---|---|
| Recall | 0.898 | **0.942** | **+0.043** |
| Precision | 0.762 | **0.799** | +0.037 |
| F1 | 0.825 | **0.865** | **+0.040** |
| recall regressions | 101 | 43 | −58 |
| seam splits | 45 | 16 | −29 |
| SAHI-unique FP | 241 | 170 | −71 |
| YOLO↔SAHI agreement | 0.727 | 0.806 | +0.079 |

**SAHI vs YOLO after the fix:** SAHI recall **0.942 > YOLO 0.906** (it finds more heads — the point of tiling). F1 gap closed from −0.056 to **−0.016**. SAHI's one remaining deficit is precision (0.799 vs 0.857) — and some of those "false positives" are likely **real heads missing from the single hand-labeled GT image** (tiling surfaces small/edge heads a labeler skips), so the true gap is an upper bound.

---

## 3. Merge-metric study — IOS vs IOU vs CONF

Motivated by a visible failure: **a small head nested in a big diagonal head's axis-aligned box gets absorbed** (the box has empty corners; the small head falls inside). Tested three merge strategies (all on the single-threshold pipeline):

| metric | v3 **IOS** | v4 IOU | v5 CONF (protect 0.6) | YOLO |
|---|---|---|---|---|
| **F1** | **0.865** | 0.816 | 0.839 | 0.881 |
| Precision | **0.799** | 0.701 | 0.740 | 0.857 |
| Recall | 0.942 | **0.976** | 0.970 | 0.906 |
| regressions | 43 | 1 | 5 | — |
| MERGE errors | 26 | 24 | 25 | — |
| splits | **16** | 81 | 66 | — |
| SAHI-unique FP | **170** | 386 | 291 | — |

- **IOU** (intersection-over-union): fixes nested heads (regressions 43→1) but **fails to merge seam fragments** (two side-by-side halves have low IOU) → splits **16→81**, FP +250, **F1 −0.049**. Bad trade.
- **CONF** (confidence-aware, ours): keep a *high-conf* contained box (distinct nested head), absorb a *low-conf* one (fragment). It recovered nested heads (regressions 43→5) but **still split badly** (16→66) and **F1 −0.025**. Why: after the single-threshold fix, **seam fragments are no longer low-confidence** (they're ≥0.35, often ≥0.6), so confidence can't separate "fragment" from "nested head."

**Conclusion: IOS (v3) is the best of the three.** The nested-head case is **geometrically and confidence-identical** to a seam fragment, so **no box-level merge metric can separate them** — the discriminating information (two separate heads vs one head's fragment) is in the **pixels, not the boxes**.

### The CONF mode is kept as opt-in
`sahi_merge: "CONF"` (with `sahi_dup_iou_threshold`, `sahi_protect_conf`) is implemented and available but **NOT the default** — it lost to IOS on FIP. Kept because it may help on **phone** data (see §4).

---

## 4. The real fix (not done) + phone relevance

The nested-head problem is the **axis-aligned-box-on-diagonal-head** limitation. The only approach with the right information is **mask-based dedup**: SAM gives each head a distinct mask, so two heads have separate masks even when their boxes nest — mask overlap separates exactly the cases box overlap cannot. Cost: reorder the pipeline so SAM runs before the dedup. **Not implemented.**

**Phone relevance:** phone images are shot **from the side**, so many wheat heads are **diagonal** in frame — the empty-corner / nested-head case will be **more common** than on FIP's overhead views. So on phone, IOS may over-merge more, and CONF or mask-based dedup may become worth it. **Decision: revisit when running SAHI on phone; for now FIP uses IOS (v3).**

---

## 5. Chosen config (production)

SAHI default = the **v3** setup:
- `conf_threshold_good_box: 0.35` (single confidence; tiles run at it)
- `sahi_merge: "NMM"`, `sahi_match_metric: "IOS"`, `sahi_match_threshold: 0.5`
- `sahi_slice_size: 1280`, `sahi_overlap_ratio: 0.3`, `sahi_full_image_pass: true`

**Net:** on FIP, SAHI is a solid, honest win over plain YOLO on recall (its intended purpose), with a small precision deficit that is partly a GT-completeness artifact. Phenotyping-grade nested-head separation is a phone-era, mask-based follow-up.
