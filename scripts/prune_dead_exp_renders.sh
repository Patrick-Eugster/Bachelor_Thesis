#!/bin/bash
# ============================================================================
# Prune train/test render PNGs of DEAD/superseded experiments down to a single
# sample pair each (keep the alphabetically-first .png in every renders/gt/overlay/
# segmentation leaf under train/ and test/, delete the rest). Models (point_cloud/),
# seg outputs (segmentation_3d/, 2DSeg/), metrics (results.json) are untouched.
#
# Usage: prune_dead_exp_renders.sh            # dry-run: report bytes that WOULD be freed
#        prune_dead_exp_renders.sh --apply    # actually delete
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
APPLY="${1:-}"

# the dead-experiment dirs (matched by the flags we agreed on)
mapfile -t EXPS < <(find results/reconstruction -maxdepth 6 -type d \( \
  -name 'pipeline_smoke*' -o -name 'test_diffgs_full' -o -name 'initial*' -o -name 'cullval*' \
  -o -name 'paper_bench_30k_pp' -o -name 'agisoft_2group_old' -o -name 'phone_sahi' \
  -o -name 'colmap_bench' -o -name 'agisoft_bench' \) 2>/dev/null \
  | grep -E '/vanilla_3dgs/[^/]+$|/agisoft_2group_old$' | sort)

echo "=== ${#EXPS[@]} dead experiments ==="
freed=0; kept=0
for E in "${EXPS[@]}"; do
  # every leaf dir that holds PNGs, under this experiment's train/ or test/
  while IFS= read -r d; do
    [ -z "$d" ] && continue
    # keep the first png (sorted), delete the rest
    first=$(find "$d" -maxdepth 1 -name '*.png' -printf '%f\n' 2>/dev/null | sort | head -1)
    [ -z "$first" ] && continue
    kept=$((kept+1))
    while IFS= read -r f; do
      [ "$(basename "$f")" = "$first" ] && continue
      sz=$(stat -c%s "$f")
      freed=$((freed+sz))
      [ "$APPLY" = "--apply" ] && rm -f "$f"
    done < <(find "$d" -maxdepth 1 -name '*.png')
  done < <(find "$E/train" "$E/test" -type d 2>/dev/null)
done

printf "kept 1 sample in %d png-dirs; %s %.1f GB\n" "$kept" \
  "$([ "$APPLY" = "--apply" ] && echo freed || echo 'WOULD free')" "$(awk "BEGIN{print $freed/1024/1024/1024}")"
[ "$APPLY" != "--apply" ] && echo "(dry-run — re-run with --apply to delete)"
