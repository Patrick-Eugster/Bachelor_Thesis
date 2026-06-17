# SAHI Mask-Based Dedup (experimental)

> **Status: EXPERIMENTAL, standalone, NOT production.** Production SAHI uses the IOS box-merge (`metrics_v3`, see [`SAHI_EVAL_RESULTS.md`](SAHI_EVAL_RESULTS.md)). This file does not touch the normal pipeline — it's a separate script + config you opt into.

## Why it exists

The production SAHI merge combines overlapping-tile boxes by **box overlap** (IOS-NMM). Box overlap cannot tell two situations apart:

| situation | boxes look like | want |
|---|---|---|
| same head detected in 2 tiles | overlapping boxes | **merge** → one |
| small head nested in a big **diagonal** head's axis-aligned box (empty corner) | overlapping boxes (small inside big) | **keep both** |

So box-IOS wrongly **absorbs** the nested head. The fix idea: decide on **mask** overlap instead — two distinct heads have distinct masks even when their boxes nest. The feasibility spike (8/10 nested pairs separated by SAM with the right prompt) is in [`SAHI_EVAL_RESULTS.md`](SAHI_EVAL_RESULTS.md) §6.

> **IoU vs IoS** (used throughout below): IoU divides the overlap by the **union** (are these the same box?); IoS divides by the **smaller** box (is the small one inside the big one?). A small box inside a big one → IoU low, IoS ≈ 1. Full reference with a worked example: [`SAHI_EVAL_RESULTS.md`](SAHI_EVAL_RESULTS.md) §0.

## How it works (flow, per image)

```
SAHI tiles ─▶ YOLO on each tile ─▶ all pre-merge boxes        (NO box-merge)
                                         │
                                         ▼  IoU pre-collapse (0.8): drop near-identical duplicate
                                            boxes, keep highest conf (cheap; nested heads have LOW
                                            IoU so they survive)
                                         │
                                         ▼  SAM one CLEAN mask per surviving box:
                                            • 👍 positive point at the box centre  ("this head")
                                            • 👎 negative points at centres of DISTINCT overlapping
                                                 neighbours ("not those")
                                         │
                                         ▼  mask dedup (greedy, conf-ordered):
                                            two masks overlapping (mask-IoS ≥ 0.6) = SAME head → merge;
                                            little overlap = distinct heads → keep both
                                         │
                                         ▼  save masks/ + bboxes/ (= bbox of each kept mask) +
                                            bboxes_with_conf/ + viz/ (coloured masks) + yolo_vis/ (boxes)
```

**The negative point is the crux.** A lone centre point makes SAM grab the biggest object touching it; a centre point **+ a negative point on the neighbour** carves the boundary between two heads sharing a rectangle. A negative must be a **distinct** neighbour — high box-IoS (contained/overlapping) **AND low box-IoU** (`neg_iou_max`), so a near-duplicate box of the *same* head isn't used as a negative (that would put a 👎 on the very head we're segmenting → garbage mask).

**Scope (important):** this only recovers heads YOLO **detected** (each detected head has a box). It does **not** find heads YOLO missed — no box → no prompt → no mask. So it fixes *merge* misses, not *detection* misses.

## Files (production untouched)

- [`src/mask_generation/sahi_yolo_sam/sahi_mask_dedup.py`](../src/mask_generation/sahi_yolo_sam/sahi_mask_dedup.py) — standalone Hydra script.
- [`configs/mask_generation/sahi_mask_dedup.yaml`](../configs/mask_generation/sahi_mask_dedup.yaml) — its config.
- Imports `compute_tile_boxes` / `load_and_slice` / `infer_tiles` / `_iou_ios` from `sahi_yolo_pipelined` **read-only** (calls them, never `merge_preds`). `run_mask_generation` / `sahi_yolo_pipelined` / `sam_v1` are unmodified.

## Knobs

| knob | default | meaning |
|---|---|---|
| `pre_collapse_iou` | 0.8 | drop near-identical boxes (IoU≥this) before SAM — fewer SAM calls; nested heads (low IoU) untouched |
| `neg_ios_min` | 0.3 | a box overlapping this one (box-IoS≥this) is a negative-point candidate |
| `neg_iou_max` | 0.5 | …but only if box-IoU<this (so a same-head duplicate isn't used as a negative) |
| `max_neg` | 5 | cap negative points per box (nearest by centre) |
| `mask_dedup_ios` | 0.6 | two masks overlapping (mask-IoS≥this) = same head → merge; below = distinct → keep |

## Run

```bash
# one labeled image (quick look)
python src/mask_generation/sahi_yolo_sam/sahi_mask_dedup.py plot_glob=plot_461 limit_images=1 labeled_only=true
# all labeled FIP images
python src/mask_generation/sahi_yolo_sam/sahi_mask_dedup.py labeled_only=true
```
Output → `results/mask_generation/{dataset}/{plot}/sahi_yolo_sam/{experiment_name}/{bboxes,bboxes_with_conf,masks,viz,yolo_vis}/`. Evaluate with the 3-way compare: `eval_compare_3way.py ... sahi_experiment=maskdedup_v1`.

## Results so far (honest — plot_461)

| | recall | precision | F1 | sahi_rescued (nested recovered) | FP |
|---|---|---|---|---|---|
| IOS v3 (production) | 0.908 | 0.762 | 0.829 | **10** | 58 |
| mask-dedup v1 | 0.872 | 0.833 | **0.852** | **5** | 38 |

- F1 is marginally *higher* than IOS — but only because it cuts false positives (38 vs 58) at a recall cost. That's a precision↔recall **rebalance**, not the intended win.
- **It fails its goal:** `sahi_rescued` went **down** (10 → 5) — it recovers *fewer* nested heads than the plain box-merge.
- **Why:** the nested corner heads are mostly a **detection** miss (YOLO never boxed them on the tiles), not a merge miss — so there's no box for mask-dedup to un-merge. (The spike worked because it used GT boxes, which always exist.)
- The box-IoU eval is also the wrong ruler — mask-dedup's real value would be **mask quality**, which a box metric can't see.

**Conclusion:** as built, this doesn't beat IOS at its actual purpose. The nested-diagonal-head problem is largely a **detector** limitation (upstream: a better/oriented-box detector or training), not a post-processing one.

## Better future approach (TODO): surgical / hybrid dedup

The v1 above runs SAM on **every** box (hundreds/image) — most of which are **non-overlapping** clean single heads that never needed it, and reprocessing them through SAM only adds noise (and is slow). The better design only spends SAM where there's genuine ambiguity:

1. **Box logic for the clean majority (no SAM):**
   - non-overlapping box → one head, keep as-is.
   - high-**IoU** overlap → same head detected twice → NMS-merge (cheap, no SAM).
2. **SAM only for the ambiguous clusters:** the **contained** pairs (high box-IoS **and** low box-IoU = nested-head-vs-fragment) — typically a small fraction of boxes (~the 26 "merge errors" on FIP). Run point+negative SAM only there to decide split-vs-keep.
3. **Evaluate on masks, not boxes:** use `eval_seg_2d` (pixel IoU vs the GT masks in `manual_label/`), since the value is mask quality.

This keeps the strong IOS baseline (0.908 recall) untouched, is far faster, and surgically targets only the nested cases — addressing v1's two failures (it disturbed the easy heads, and it was slow). It's also a **smaller build than v1**: reuse the IOS box-merge as the base and add a SAM check *only* for the contained pairs.

### Why v1 was wrong to SAM everything
v1 runs SAM on **100%** of boxes. But most boxes are **non-overlapping** — a head detected once, no neighbour, already correct. Running SAM on those is pure waste **and adds noise** (which is part of why v1 made the easy heads slightly *worse*). SAM should be spent **only where there's genuine ambiguity** (~10–20% of boxes). The surgical version would run SAM on a fraction of the boxes, much faster, leaving the clean majority alone.

### ⚠️ Verify BEFORE building it (the gate)
Even surgical, this can only **un-merge heads YOLO detected** — it cannot recover a head with no box. The plot_461 eval pointed at the nested corner heads being mostly a **detection miss** (YOLO never boxed them on the tiles), which puts a **low ceiling** on how much *any* dedup can recover. So **the first step is a cheap check, not code:** confirm the nested corner heads actually appear in SAHI's *pre-merge* YOLO boxes (e.g. extend `sahi_merge_debug.py` to dump pre-merge boxes and overlay the GT nested pairs). If they're there → build the surgical version. If they're absent → the problem is upstream in the **detector** (a better/oriented-box detector, or training), and post-processing can't fix it — don't spend the effort.

**Bottom line:** the nested-diagonal-head problem is ultimately a **detector** limitation; surgical mask-dedup is the *most* it's worth spending on the post-processing side, and only after the verify-first gate passes.
