#!/bin/bash -l
#SBATCH -J phone_metrics_rerun
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G       # 36 GB total (6G x 6 cpus) — generous headroom, metrics itself is light
#SBATCH --cpus-per-task=6
#SBATCH --time=03:59:00        # 32 arms x ~1-2 min = ~30-65 min; padded to strictly under 4h (short tier)
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_metrics_rerun_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_metrics_rerun_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ============================================================================
# PHONE METRICS-ONLY RERUN — fills the region gaps in results.json UNIFORMLY across all 32 arms.
# ----------------------------------------------------------------------------
# WHY: results.json currently has `whole` (PSNR/SSIM/LPIPS/sharpness) on all arms, but the region passes
#   are inconsistent — `inner` only on the Group-X radial arms, `roi`+`markers` only on the pinhole arms
#   (+fD/0627 agisoft). The opencv/radial/agisoft arms were metric'd BEFORE their marker_points3d.json was
#   generated (2026-08-13), so ROI/MARKERS silently skipped. This re-runs metrics.py on every arm now that
#   markers exist, so inner+roi+markers (all carrying LPIPS) land uniformly. NO retrain — reads test/renders.
#
# ⚠️ PREREQUISITE: the opencv/radial/agisoft arms need logs/marker_points3d.json on Euler (generated LOCALLY
#   Aug 13 -> push them up first with the rsync in scripts/push_marker_jsons_to_euler.sh). The preflight
#   below COUNTS how many arms have their marker file and prints which are missing — it does NOT abort
#   (a missing marker file just means that arm keeps whole+inner only, same as before), so it's safe to run.
#
# ADDITIVE + SAFE: each results.json is backed up to results.json.prebak first; on a per-arm failure the
#   backup is restored. whole/inner recompute identically; only the previously-skipped regions get added.
# ============================================================================

REPO=/cluster/project/cropsci/peugste/wheat3dgs
module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy
cd "$REPO"
nvidia-smi

LOG="$REPO/slurm_logs/phone_metrics_rerun_${SLURM_JOB_ID}.log"
: > "$LOG"

# collect every phone baseline arm that has a results.json (= every trained/metric'd arm)
mapfile -t ARMS < <(find results/reconstruction/phone -path '*/baseline/results.json' | sort)
echo "found ${#ARMS[@]} phone baseline arms" | tee -a "$LOG"

# ── PREFLIGHT: report marker-file coverage (ROI/MARKERS need it; missing -> that arm keeps whole+inner) ──
echo "=== PREFLIGHT: marker_points3d.json coverage per arm ===" | tee -a "$LOG"
have=0; miss=0
for rj in "${ARMS[@]}"; do
  model=$(dirname "$rj")
  src=$(python3 - "$model" <<'PY'
import sys, os
m=sys.argv[1].split('/')                # results reconstruction phone <f> <d> [<variant>] vanilla_3dgs baseline
f,d=m[3],m[4]; vi=m.index('vanilla_3dgs')
variant=m[vi-1] if m[vi-1]!=d else ''
print(os.path.join('input_plots/phone',f,d,variant))
PY
)
  if [ -f "$src/logs/marker_points3d.json" ]; then have=$((have+1));
  else miss=$((miss+1)); echo "  (no markers -> whole+inner only): $src" | tee -a "$LOG"; fi
done
echo "marker file present on $have arms, missing on $miss" | tee -a "$LOG"

# ── RUN metrics.py per arm (metrics-only; -s enables ROI+MARKERS when the marker file exists) ──
declare -A STATUS
for rj in "${ARMS[@]}"; do
  model=$(dirname "$rj")
  src=$(python3 - "$model" <<'PY'
import sys, os
m=sys.argv[1].split('/')
f,d=m[3],m[4]; vi=m.index('vanilla_3dgs')
variant=m[vi-1] if m[vi-1]!=d else ''
print(os.path.join('input_plots/phone',f,d,variant))
PY
)
  echo "" | tee -a "$LOG"
  echo "==== $(date '+%H:%M:%S')  $model   (src=$src) ====" | tee -a "$LOG"
  cp -f "$rj" "$rj.prebak"                                  # safety backup before overwrite
  if python src/reconstruction/metrics.py -m "$model" -s "$src" >>"$LOG" 2>&1; then
    summary=$(python3 -c "
import json
d=json.load(open('$rj')); k=sorted(d)[-1]; m=d[k]
def g(r): return ('%s PSNR %.2f'%(r,m[r]['PSNR'])) if m.get(r) else '%s -'%r
print('  OK  ' + ' | '.join(['whole PSNR %.2f'%m['PSNR'], g('inner'), g('roi'), g('markers')]))
" 2>/dev/null)
    echo "${summary:-  OK (summary parse failed)}" | tee -a "$LOG"
    STATUS["$model"]=OK
  else
    echo "  FAIL (restoring backup)" | tee -a "$LOG"; cp -f "$rj.prebak" "$rj"
    STATUS["$model"]=FAIL
  fi
done

# ── final status table ──
echo "" | tee -a "$LOG"
echo "================ PER-ARM STATUS ================" | tee -a "$LOG"
ok=0; fail=0
for rj in "${ARMS[@]}"; do
  model=$(dirname "$rj"); st=${STATUS[$model]:-?}
  [ "$st" = OK ] && ok=$((ok+1)) || fail=$((fail+1))
  echo "  $st  $model" | tee -a "$LOG"
done
echo "-----------------------------------------------" | tee -a "$LOG"
echo "OK=$ok  FAIL=$fail  of ${#ARMS[@]}" | tee -a "$LOG"
echo "PULL BACK (from ~, results.json only): rsync the '*/baseline/results.json' files, exclude renders/point_cloud/chkpnt." | tee -a "$LOG"
