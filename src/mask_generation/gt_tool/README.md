# Point-GT tool

Interactive local tool to create **pixel mask ground truth** for phone wheat heads with **SAM2.1 point
prompts**. Browser-based, zero new deps (stdlib HTTP server). Design + rationale + the two important bugs:
[../../../docs/mask_generation/POINT_GT_TOOL.md](../../../docs/mask_generation/POINT_GT_TOOL.md).

## Run

```bash
# in the dev container (base env: torch + ultralytics)
python -m mask_generation.gt_tool.server
# open http://localhost:8000 in the Windows browser
```
Env: `GT_SAM_WEIGHT` (default `sam2.1_l.pt`), `GT_DECODE_BATCH` (8), `GT_PORT` (8000).

> **Backend change → restart the server** (`Ctrl+C` + rerun). **Frontend change → hard-reload** the page
> (`Ctrl+Shift+R`). The page is served fresh each load; `server.py` is loaded once at startup.

## Workflow

1. Pick a GT image in the left list. With **`⚙ auto-seed` OFF (default)** it opens **blank**; with it ON (or
   after **`＋ seeds`**) it opens with the YOLO+SAM masks. Saved work always reloads.
2. **Label heads** (see controls): click a head → `E` → drop **⊕ positive** points on it (and **⊖ negative**
   on neighbours it bleeds into) → **`Enter` / ▶ Run SAM** to segment. A toast confirms `✓ SAM done — N px`.
   For a head with no mask, just click empty ground in edit mode (or `N`). After Run, SAM offers **3
   candidate masks** — the best-fitting one is auto-shown; press **`Tab`** (or the `⟳ i/3` button) to cycle to
   the other two and keep whichever looks right.
   - **When SAM can't get it (overlapping heads, low-res/compressed):** don't fight the points. Use the
     **manual tools** — `F` **Brush** to paint/erase pixels straight onto the selected mask (fix SAM's
     spillover or add a missing tip; Alt-drag erases), or `G` **Polygon** to trace a clean new head from
     scratch. Both **accumulate a live preview and only write the binary mask when you press `Enter`** — so
     you draw freely and nothing commits mid-stroke. A hand-drawn head is identical on disk to a SAM one.
3. **Fix seeds** (if seeded): select a loose mask → `E` → a couple points → Run; `Del` false ones; `L` lock
   the good ones.
4. **💾 Save** → writes into `input_plots/phone/<field>/<date>/manual_label/` (`<stem>_gt_mask.png` = union
   of the **active set** = the eval GT; plus `<stem>_sets/` with every set; `<stem>_meta.json`).

## Controls

| action | how |
|---|---|
| **select** a mask | left-click (SELECT mode) |
| **points** edit selected (SAM prompts) | `E` / `✎ Points` (glows red) |
| **⊕ positive / ⊖ negative** point | left-click in edit / **`Q`** swaps · **Shift-click = +** · **Alt-click = −** |
| **delete a placed point** | click on it (it glows on hover) |
| **brush** paint/erase the selected mask | `F` / `🖌 Brush` — drag = paint, **Alt-drag = erase**; set size via the toolbar **slider / px field** (or `[` `]` on US layouts) |
| **polygon** draw a new head from scratch | `G` / `⬡ Polygon` — click vertices, `Backspace` undo vertex |
| **commit the edit** | `Enter` / the commit button (Run SAM / Apply brush / Create head) |
| **cycle SAM's 3 candidate masks** | `Tab` / `Shift+Tab` / the `⟳ i/3` button (after Run SAM) |
| **new head** | click empty ground in edit, or `N` then click |
| back to select | `↖ Select` / `Esc` |
| delete selected mask | `Del` |
| lock / unlock (protect + survive hide-all) | `L` / `🔒` |
| hide selected mask | `X` |
| show/hide overlay | `M` / `👁 masks` |
| hide all but locked | `H` / `👁 hide all` |
| background: photo → black → white | `V` / `🎞 bg` (black/white = mask-image view, masks shown solid) |
| mask colour: per-id colour / binary white | `C` / `🎨` (colour = separate heads by colour; binary = clean white mask) |
| ROI border | `R` / `▦ ROI` |
| pan | **WASD** (Shift = faster) · **right-drag** · Space+drag · Shift+wheel (horizontal) |
| zoom | **wheel** (no modifier — to cursor) · `+` / `-` |
| prev / next image | `,` / `.` |
| sidebar | `B` / `☰` |
| **save** | `Ctrl+S` / `💾 Save` |

## Mask-sets (versions) + safe clearing

- **`set:` dropdown + `＋`** (top bar) — keep multiple mask-sets per image; switch between them, add a new
  empty one. The **active** set is edited and saved.
- **`＋ seeds`** — load the YOLO+SAM masks into the active set (**appends**, never deletes).
- **`🗑 clear`** (click to arm → click to confirm) — moves the set's masks into a **`⟲ set N` backup set**
  (kept in the dropdown; locked masks stay).
- **`💾 Save` persists ALL sets to disk**, so sets + `⟲` backups **survive reload**. The GT that eval reads is
  the **active set at save time** — switch to the set you want as GT before saving.

## Notes

- **SAM2.1-large**, per-head crop (the resolution lever). First seed of an image ~48 s (cached; reopen fast);
  each point-click is sub-second.
- Single active image / single user; SAM calls serialized on one GPU (~6 GB interactive).
- This is the **mask-GT** track (feeds `eval_seg_2d`). The **box-GT** track (CVAT → `eval_yolo_boxes`) is
  separate — see `../../../docs/mask_generation/PHONE_GT_LABELING.md`.
