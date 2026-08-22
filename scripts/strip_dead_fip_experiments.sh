#!/bin/bash
# ============================================================================
# Strip HEAVY content from dead/superseded FIP reconstruction experiments, keeping only the KB-sized record
# (results.json, config.yaml, cfg_args, run_report, cameras.json, per_view.json). Deletes the 3DGS model
# (point_cloud/), seg label maps (2DSeg/*.pt), all *.ply, and leftover render *.png — all disposable since
# we never re-seg/re-render a dead run. The experiment folder survives as a lightweight breadcrumb.
#
# Usage: strip_dead_fip_experiments.sh          # dry-run: report bytes that WOULD be freed
#        strip_dead_fip_experiments.sh --apply   # actually delete
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
APPLY="${1:-}"

# Tier A: dead / smoke / known-bad, results already recorded (see FIP_SEG_PP_ABSGRAD_AB.md + memory)
EXPS=(test_diffgs_full cullval_absgrad paper_bench_30k_pp initial initial_30k_iterations pipeline_smoke_461 gsplat_smoke)

freed=0; nfiles=0; ndirs=0
for e in "${EXPS[@]}"; do
  for d in results/reconstruction/fip/plot_46[1-7]/vanilla_3dgs/"$e"; do
    [ -d "$d" ] || continue
    # heavy files: models, seg maps, plys, renders
    while IFS= read -r f; do
      sz=$(stat -c%s "$f"); freed=$((freed+sz)); nfiles=$((nfiles+1))
      [ "$APPLY" = "--apply" ] && rm -f "$f"
    done < <(find "$d" \( -name '*.ply' -o -name '*.pt' -o -name '*.png' -o -name '*.pth' \) -type f 2>/dev/null)
    # remove now-empty point_cloud/ skeletons (and any other emptied dirs), apply-only
    if [ "$APPLY" = "--apply" ]; then
      find "$d" -type d -empty -delete 2>/dev/null || true
    fi
  done
done

printf "%s %.1f GB across %d files (7 dead experiments)\n" \
  "$([ "$APPLY" = "--apply" ] && echo 'FREED' || echo 'WOULD free')" "$(awk "BEGIN{print $freed/1073741824}")" "$nfiles"
echo "  kept: results.json / config.yaml / cfg_args / run_report / cameras.json / per_view.json (the record)"
[ "$APPLY" != "--apply" ] && echo "  (dry-run — re-run with --apply to delete)"
