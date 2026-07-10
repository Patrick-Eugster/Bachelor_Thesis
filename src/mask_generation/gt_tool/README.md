# Point-GT tool

Interactive tool to create **pixel mask ground truth** for phone wheat heads by correcting a SAM draft
with point prompts. Local, browser-based, zero new dependencies (stdlib HTTP server). Design + rationale:
[../../../docs/mask_generation/POINT_GT_TOOL.md](../../../docs/mask_generation/POINT_GT_TOOL.md).

## Run

```bash
# in the dev container (base env: torch + ultralytics already there)
python -m mask_generation.gt_tool.server
# then open http://localhost:8000 in your Windows browser
```

Config via env vars (optional):
- `GT_SAM_WEIGHT` — SAM checkpoint (default `sam2.1_l.pt` in the repo root)
- `GT_DECODE_BATCH` — seed-pass batch size (default `8`; controls the VRAM of the one-time auto-seed)
- `GT_PORT` — port (default `8000`)

> **After restarting the server, hard-reload the tab (`Ctrl+Shift+R`).** The server keeps its state in
> memory and resets on restart; a stale tab and a fresh server disagree until you reload.

## Workflow

1. Pick a GT image in the left list. **First time only**, the server SAM-seeds every YOLO/SAHI box into a
   draft mask (~200–300 heads, ~20–30 s — see *Caching* below). The **image shows immediately**; the masks
   fill in when the seed finishes.
2. **Fix the bad heads:** left-click a mask to **select** it, then **left-click = positive point**,
   **right-click = negative point** (put negatives on the occluding neighbour). The mask re-segments on
   each click, on a tight per-head crop.
3. **N** starts a **new** head (next left-click seeds it); keep clicking to add points.
4. **Del** deletes the selected instance. **Esc** deselects.
5. **Ctrl+S** saves → writes into `input_plots/phone/<field>/<date>/manual_label/`:
   `<stem>_gt_mask.png` (binary union, for `eval_seg_2d`), `<stem>_instances.png` (uint16 instance map),
   `<stem>_seed.json` (per-head boxes/points — resumable + replayable through another SAM), `<stem>_meta.json`.

## Reading the UI

### Left sidebar
- Grouped by **session** (`field_A/20250618`), with a divider bar and a **`done/total`** count per session.
- Per-image status icon:
  | icon | meaning |
  |---|---|
  | **`•`** grey | never opened — the one-time ~30 s seed will run when you open it |
  | **`◐`** amber ("half moon") | **seeded & cached** — reopens **instantly** from disk, no re-decode |
  | **`✔`** green | **saved GT** exists (`<stem>_gt_mask.png` written) |
- **`☰`** (top-left) or the **`B`** key hides/shows the sidebar to give the canvas full width.

### Top status bar
- **image path** — `field/date/stem` of what's loaded.
- **instances: N** — how many head masks are currently in the image.
- **sel: #id (Npt)** — the selected instance and how many points you've placed on it (`none` if nothing selected).
- **mode:** `SELECT` (clicks pick a head) · `EDIT` (a head is selected, clicks add points to it) · `NEW`
  (next click starts a new head).
- **`● unsaved`** (orange) — you have edits not yet written to disk. `Ctrl+S` clears it. Switching image or
  closing the tab while this shows will warn you first.
- **`⏳ seeding N heads…`** — the one-time SAM seed is running (image is already visible during it).
- **`ROI`** button (or **`R`** key) — toggles a **cyan dashed outline** of the plot's marker-hull region
  (the same area the pipeline greyed out when making the seed boxes), so you can see which heads are
  in-plot. Lights up blue when on. If the plot's markers weren't all triangulated, there's no ROI and it
  says so instead of drawing a partial border.

## Caching & resume (why re-opening is fast)

The SAM decode of all ~200–300 masks happens **once per image**. After the seed, the draft is written to
`input_plots/phone/<field>/<date>/gt_cache/` (`<stem>_instances.png` + `<stem>_seed.json`). From then on:
- **Re-opening a seeded image** reloads from that cache in ~0.2 s — no GPU, no wait (the `◐` icon marks these).
- **Re-opening a saved image** restores your **saved corrections** (from `manual_label/`), not the raw seed.
- Only a **never-opened** image (`•`) pays the one-time seed.

Notes:
- The cache lives under `input_plots/` (already gitignored) — not committed.
- If you regenerate the seed boxes, delete that image's entry in `gt_cache/` to force a fresh seed.
- **Unsaved** point-edits live only in memory: switching image / closing the tab warns you; if you discard,
  you get the cached draft (or last save) back, never your discarded edits.

## Controls

| | |
|---|---|
| select / +point | left-click |
| −point | right-click |
| new head | `N` then left-click |
| delete selected | `Del` |
| undo last point | `Ctrl+Z` |
| deselect | `Esc` |
| save | `Ctrl+S` |
| pan | wheel (vert), Shift+wheel (horiz), Space+drag |
| zoom | Alt+wheel (or Ctrl+wheel), `+` / `-` |
| prev / next image | `,` / `.` |
| show/hide sidebar | `B` (or the `☰` button) |
| toggle ROI border | `R` (or the `ROI` button) |

## Notes

- **Draft model = SAM2.1-large** by default; the per-head crop (not the model) is the resolution lever
  for tiny/overlapping heads. Swap the draft with `GT_SAM_WEIGHT`.
- Single active image / single user by design; SAM calls are serialized on one GPU. Loading an image bursts
  VRAM to ~12 GB during the seed, then drops to ~1–2 GB; each point-click after that is sub-second.
- This is the **mask-GT** track (feeds `eval_seg_2d`). The **box-GT** track (CVAT → `eval_yolo_boxes`)
  is separate and unchanged.
- Phase-1 MVP scope: select / point-refine / new / delete / undo-point / save / cache+resume / ROI border /
  collapsible sidebar. Not yet: full op-stack undo of deletes, box-drag new instance, opacity/hide toggles.
