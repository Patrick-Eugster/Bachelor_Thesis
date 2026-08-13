#!/usr/bin/env bash
# Re-run metrics.py on every phone baseline arm so the new `inner` (crop-fair centered-80%) region gets
# computed, alongside the existing whole/roi/markers. ADDITIVE: whole/roi are recomputed identically; only
# `inner` is added. Backs up each results.json to results.json.prebak first (safety, given the overwrite).
set -u
cd /workspace
LOG=/workspace/scripts/rerun_metrics_inner_phone.log
: > "$LOG"
mapfile -t ARMS < <(find results/reconstruction/phone -path '*/baseline/results.json' | sort)
echo "found ${#ARMS[@]} phone baseline arms" | tee -a "$LOG"
for rj in "${ARMS[@]}"; do
  model=$(dirname "$rj")                                   # .../<variant>/vanilla_3dgs/baseline
  # derive the SfM source: input_plots/phone/<f>/<d>[/<variant>]
  src=$(python3 - "$model" <<'PY'
import sys, os
m=sys.argv[1].split('/')                # results reconstruction phone <f> <d> [<variant>] vanilla_3dgs baseline
f,d=m[3],m[4]; vi=m.index('vanilla_3dgs')
variant=m[vi-1] if m[vi-1]!=d else ''
print(os.path.join('input_plots/phone',f,d,variant))
PY
)
  echo "==== $(date '+%H:%M:%S')  $model   (src=$src) ====" | tee -a "$LOG"
  cp -f "$rj" "$rj.prebak"                                 # safety backup
  if python src/reconstruction/metrics.py -m "$model" -s "$src" >>"$LOG" 2>&1; then
    inner=$(python3 -c "import json;d=json.load(open('$rj'));k=sorted(d)[-1];print('inner PSNR %.2f'%d[k]['inner']['PSNR'] if d[k].get('inner') else 'inner MISSING')" 2>/dev/null)
    echo "  OK  $inner" | tee -a "$LOG"
  else
    echo "  FAIL (restoring backup)" | tee -a "$LOG"; cp -f "$rj.prebak" "$rj"
  fi
done
echo "==== DONE $(date '+%H:%M:%S') ====" | tee -a "$LOG"
