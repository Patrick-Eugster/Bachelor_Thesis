# SAHI Surgical / Hybrid Dedup (experimental)

> **Status: EXPERIMENTAL, standalone, NOT production.** Production SAHI still uses the IOS box-merge
> (`metrics_v3`, see [`SAHI_EVAL_RESULTS.md`](SAHI_EVAL_RESULTS.md)). This is a **separate** script +
> config — it does not touch `run_mask_generation`, `sahi_yolo_pipelined`, `sam_v1`, or the v1
> mask-dedup ([`SAHI_MASK_DEDUP.md`](SAHI_MASK_DEDUP.md)). It is the "better future approach" that the
> v1 doc flagged as a TODO, now built.

## The core difference from v1

The v1 mask-dedup runs SAM on **100%** of the boxes, then dedups on masks. But most boxes are
**non-overlapping** — a head detected once, no neighbour, already correct. Running SAM on those is
pure waste **and adds noise** (part of why v1 made the easy heads slightly *worse*). SAM should only
be spent where there's genuine ambiguity.

So surgical flips it: **box logic decides the clean majority; SAM only touches the ambiguous
contained pairs.** SAM becomes a *tie-breaker*, not the mask producer. The final product is a **box
set** (same as production's YOLO phase) — the actual SAM masks still come later in the normal SAM
phase. That separation is the key cleanup: v1 conflated "use SAM to decide" with "produce final
masks"; surgical keeps them apart.

| | v1 mask-dedup | surgical |
|---|---|---|
| SAM runs on | **every** box (100%) | **only** contained pairs (~10–20%) |
| clean single heads | re-segmented (noise) | **untouched** (kept as YOLO box) |
| same-head duplicates | mask overlap | cheap **IoU-NMS** (no SAM) |
| SAM's role | produce the masks | **tie-break** split-vs-keep on nested pairs |
| speed | slow (SAM ×N) | fast (SAM ×few) |
| baseline recall | disturbed | **preserved by construction** |

## Flow per image (the 3 tiers)

```
SAHI tiles ─▶ YOLO per tile ─▶ all pre-merge boxes      (reuse load_and_slice + infer_tiles, read-only)
                                      │
                                      ▼  TIER 2 first: IoU-NMS collapse (IoU ≥ nms_merge_iou)
                                         = same head detected in 2 tiles → keep max-conf, drop rest.
                                         Nested heads have LOW IoU → they SURVIVE this step.
                                      │
                                      ▼  classify each survivor via box overlap (_iou_ios vs all):
          ┌───────────────────────────┴───────────────────────────┐
          ▼ TIER 1  (no SAM)                                       ▼ TIER 3  (SAM)
   no contained partner                              has a contained partner:
   = clean single head                               box-IoS ≥ contained_ios AND box-IoU < contained_iou_max
   → keep the YOLO box as-is                         = nested-head-vs-fragment, AMBIGUOUS
          │                                                         │
          │                                          ▼ SAM one mask per ambiguous box, BOX + point +
          │                                            negatives (box bounds the mask → no leak; negs
          │                                            carve distinct neighbours out)
          │                                          ▼ area guard: mask bbox > max_area_ratio × det box
          │                                            = leaked → drop mask, keep raw YOLO box (🟠)
          │                                          ▼ decide on MASK overlap (greedy, conf-ordered):
          │                                            mask-IoS ≥ decide_mask_ios → SAME head → drop;
          │                                            below → DISTINCT head → keep both
          └───────────────────────────┬───────────────────────────┘
                                       ▼  final box set = tier-1 boxes + tier-3 survivors
                                          save bboxes/ + bboxes_with_conf/ + yolo_vis/ (+ viz/)
                                          yolo_vis colours: 🟢 tier-1 kept · 🟣 tier-3 SAM mask · 🟠 tier-3 leaked→raw box
```

**Why the negative point matters (same as v1):** a lone centre point makes SAM grab the biggest
object touching it; a centre point **+ a negative point on the neighbour** carves the boundary
between two heads sharing a rectangle. A negative must be a **distinct** neighbour — high box-IoS
(contained/overlapping) **AND** low box-IoU (`neg_iou_max`), so a near-duplicate of the *same* head
isn't used as a negative (that would put a 👎 on the very head we're segmenting → garbage mask).

## Tier-3 leak fix (box prompt + area guard)

The first build prompted SAM with **points only** (positive centre + negatives). A point lets SAM
grab an oversized region, so tier-3 produced three visible failures: a magenta box around **several
heads**, a box over **nonsense** (not a head), and a **giant box over no head at all**. All three are
the same thing — the SAM mask **leaked** off the head, and we output the leaked mask's bounding box.
Two changes (in this file only — v1's `sam_clean_mask` is untouched) fix it:

1. **Box + point + negative prompt** (`use_box_prompt: true`, `sam_box_mask`). SAM also takes the
   **detection box** as a prompt. The box **bounds** the mask to that head's rectangle, so SAM
   physically cannot grab a neighbour or background far outside it — this kills the "giant empty box"
   and "nonsense blob". The negatives still carve a distinct neighbour out *within* the box. (The
   spike only ever tried box-alone / point-alone / point+neg — never **box+point+neg**, which is SAM's
   strongest mode and exactly what this case needs.)
2. **Area guard** (`max_area_ratio: 1.5`). If a mask's bbox area is still > 1.5× its detection-box
   area, it **leaked** → discard the mask and **fall back to the raw YOLO box** for that head (so the
   head isn't lost, it just isn't trusted as a SAM mask). Set `max_area_ratio: 0` to disable.

In `yolo_vis` these fallbacks are drawn **🟠 orange** (raw box kept, SAM rejected), distinct from
🟢 tier-1 (kept as-is) and 🟣 tier-3 (SAM mask accepted) — so a glance tells you where SAM leaked.

## Files (production AND v1 untouched)

- [`src/mask_generation/sahi_yolo_sam/sahi_surgical_dedup.py`](../src/mask_generation/sahi_yolo_sam/sahi_surgical_dedup.py) — standalone Hydra script.
- [`configs/mask_generation/sahi_surgical_dedup.yaml`](../configs/mask_generation/sahi_surgical_dedup.yaml) — its config.
- Imports `load_and_slice` / `infer_tiles` / `_iou_ios` from `sahi_yolo_pipelined` **read-only**, and
  the pure helpers (`box_center`, `crop_mask`, `mask_ios`, `sam_clean_mask`, `iou_precollapse`,
  `mask_dedup`, `select_images`, `load_yolo`) from `sahi_mask_dedup` (v1) **read-only**. Never calls
  `merge_preds`. Nothing in production or v1 is modified.

## Knobs

| knob | default | meaning |
|---|---|---|
| `nms_merge_iou` | 0.5 | TIER 2: IoU ≥ this = same head in 2 tiles → NMS-merge (drop dup), no SAM. Nested heads (low IoU) survive. |
| `contained_ios` | 0.45 | TIER 3 gate: a survivor with box-IoS ≥ this vs another survivor is "contained" (kept v4 value) |
| `contained_iou_max` | 0.5 | TIER 3 gate: …AND box-IoU < this (so it's a nested pair, not a tier-2 duplicate) |
| `use_box_prompt` | true | TIER 3 SAM: prompt with the detection **box** + point + negatives so the mask is bounded to the head (stops leaks). false = old point-only |
| `max_area_ratio` | 1.5 | area guard: mask bbox > this × detection-box area = leaked → drop mask, keep raw YOLO box. 0 disables |
| `neg_ios_min` / `neg_iou_max` | 0.3 / 0.5 | negative-point selection for SAM (same as v1) |
| `max_neg` | 5 | cap negative points per prompt (nearest by centre) |
| `decide_mask_ios` | 0.45 | TIER 3 decision: two masks overlapping (mask-IoS ≥ this) = same head → drop; below = distinct → keep both (kept v4 value) |

## Run

```bash
# one labeled image (quick look)
python src/mask_generation/sahi_yolo_sam/sahi_surgical_dedup.py plot_glob=plot_461 limit_images=1 labeled_only=true
# all labeled FIP images
python src/mask_generation/sahi_yolo_sam/sahi_surgical_dedup.py labeled_only=true
```
Output → `results/mask_generation/{dataset}/{plot}/sahi_yolo_sam/{experiment_name}/{bboxes,bboxes_with_conf,masks,viz,yolo_vis}/`.

**Experiment names** (own folder each, for clean A/B):
- `surgical_v1` — first build, **point-only** prompt (leaky tier-3).
- `surgical_v2` — **box + point + negative** prompt + **area guard** (the leak fix). `contained_ios=0.6 decide_mask_ios=0.6`.
- `surgical_v3` — `decide_mask_ios=0.45` (merge tier-3 harder).
- `surgical_v4` — `contained_ios=0.45 decide_mask_ios=0.45` — **kept config** (best merges/splits, top recall).

A/B: `eval_compare_3way.py ... sahi_experiment=surgical_v4` vs `surgical_v2/v3`, `maskdedup_v1` (v1) and
`metrics_v3` (production IOS).

## Results — FIP benchmark (7 GT images, 1386 heads, IoU 0.35)

The leak fix (v1→v2) and the knob sweep (v2→v3→v4):

| method | recall | precision | F1 | merges | splits | FP (uniq) | boxes |
|---|---|---|---|---|---|---|---|
| plain YOLO | 0.906 | **0.857** | **0.881** | 16 | 24 | **209** | 1465 |
| SAHI IOS v3 (production) | 0.942 | 0.799 | 0.865 | 26 | 16 | 328 (170) | 1633 |
| SAHI maskdedup_v1 (SAM-everything) | 0.832 | 0.785 | 0.808 | 47 | 37 | 316 | 1469 |
| surgical_v2 (point→box prompt + guard; ci.6/dm.6) | **0.961** | 0.759 | 0.848 | 17 | 23 | 422 (236) | 1754 |
| surgical_v3 (dm.45) | **0.961** | 0.762 | 0.850 | 16 | 19 | 415 (231) | 1747 |
| **surgical_v4 (ci.45/dm.45) — kept** | 0.959 | 0.765 | 0.851 | **15** | 16 | 409 (227) | 1738 |

**Read of the numbers:**
1. **Surgical decisively beats the v1 mask-dedup** (the reason it was built): F1 0.851 vs 0.808, recall 0.959 vs 0.832, merges 15 vs 47. "SAM only the ambiguous pairs" ≫ "SAM everything" (v1 SAMmed 100% of boxes, leaky masks *swallowed* 166 heads YOLO had found).
2. **The over-merge bug is solved.** SAHI merges: IOS 26 → surgical **15** (below even plain YOLO's 16). That was the whole point.
3. **Best recall of all methods (0.959–0.961).**
4. **But IOS v3 keeps the better F1 (0.865 vs 0.851)** — purely on precision. Surgical carries ~80 extra FPs.
5. **The knob sweep converged.** Both `decide_mask_ios` (0.6→0.45) and `contained_ios` (0.6→0.45) moved precision by <0.01. **Proof the FP gap is structural, not a tuning miss:** IOS's IoS-NMM absorbs partial/seam fragments into their parent box; surgical's box logic keeps them as separate boxes (the ~57 extra *sahi-unique* FPs, 227 vs 170). No threshold touches that without re-introducing the over-merge surgical exists to remove.

**Conclusion (characterized trade-off, not a failure):**
- **Surgical = recall + merge-fix champion** (recall 0.959, merges 15). Best tool for **phone** (dense, side-view, many diagonal heads → the over-merge is worst there, and recall is what matters).
- **IOS v3 = precision/F1 champion** (F1 0.865). Stays **production for FIP**, where heads aren't dense enough for recall to dominate.
- Knobs exhausted; further FIP F1 would need a different mechanism (e.g. an IoS-NMS fragment pre-pass before tier-1), which trades back toward IOS — not worth it. `surgical_v4` is the kept config.

## The honest bound (unchanged)

Still only un-merges heads YOLO **detected** (each tier-3 box exists because YOLO boxed it). If the
nested heads aren't in the pre-merge boxes at all, tier-3 has nothing to split → the recovery ceiling
stays low and it's a **detector** limitation, not a post-processing one. We skipped the verify-gate
(see [`SAHI_MASK_DEDUP.md`](SAHI_MASK_DEDUP.md) "⚠️ Verify BEFORE building it") by choice, so we read
that ceiling directly off the `surgical_v1` vs `maskdedup_v1` vs `metrics_v3` numbers instead of
measuring it up front.
