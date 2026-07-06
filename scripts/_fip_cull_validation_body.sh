# ------------------------------------------------------------------------------
# Shared body for the FIP seg-optimization losslessness validation (sourced by the
# fip_cull_validation_{a,b,c,d}_job.sh wrappers, which set $PLOTS + the SBATCH header).
#
# GOAL: prove our seg optimizations are BIT-IDENTICAL to the original code on FIP — on
# EVERY plot, not just plot_461. Per plot we do a CLEAN train, then two seg runs on that
# same model that differ ONLY by the optimization flags, and md5-compare them:
#
#   Seg A (baseline)  : --no_mask_cache --no_frustum_cull   (original code path)
#   Seg B (optimized) : crop cache + disk cache + lift-from-crop + frustum cull + inference
#
# The gate is compare_seg_runs.py: same all_obj_labels.pth md5 AND same 2DSeg maps => lossless.
# Same model + same seg_seed(0) both sides, so the ONLY variable is the optimization flags —
# this is the clean A/B the old seg_cropcache comparison could not be (different code/order).
#
# Train arm = gsplat + use_principal_point + AbsGrad (matches scripts/fip_test_absgrad_job.sh).
# Because the model is trained pp=true, BOTH seg runs pass use_principal_point=true (FIP footgun).
# ------------------------------------------------------------------------------

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

EXP=cullval_absgrad                          # clean model folder for this validation
THRESH=0.0008                                # AbsGS densify_grad_threshold (gsplat absgrad)
PP=reconstruction.use_principal_point=true   # model trained pp=true -> seg MUST match
# Crop-cache build RAM guard (Seg B builds the cache); harmless otherwise.
export MALLOC_ARENA_MAX=2

SUMMARY=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/cullval_summary_${SLURM_JOB_ID}.txt
: > "$SUMMARY"

for PLOT in $PLOTS; do
  echo ""
  echo "################################################################"
  echo "#  ${PLOT}"
  echo "################################################################"

  # ---- 1) clean train: gsplat + pp + AbsGrad (15k — enough for seg validation) ----
  echo "==================== ${PLOT}  TRAIN (gsplat + pp + AbsGrad 15k) ===================="
  date
  python src/run_reconstruction.py \
    plot=${PLOT} \
    run_train=true run_render=false run_metrics=false \
    run_seg=false run_eval=false run_eval_2d=false \
    reconstruction.iterations=15000 \
    reconstruction.use_principal_point=true \
    reconstruction.absgrad=true \
    reconstruction.densify_grad_threshold=${THRESH} \
    experiment_name=${EXP}

  # ---- 2) Seg A — BASELINE (all optimizations OFF) ----
  echo "==================== ${PLOT}  SEG A baseline (no cache, no cull) ===================="
  date
  python src/run_reconstruction.py \
    plot=${PLOT} experiment_name=${EXP} ${PP} \
    run_seg=true run_eval=false run_eval_2d=false \
    segmentation_3d.use_mask_cache=false \
    segmentation_3d.frustum_cull=false \
    segmentation_3d.exp_name=seg_baseline

  # ---- 3) Seg B — OPTIMIZED (cache + lift-from-crop + cull + inference) ----
  echo "==================== ${PLOT}  SEG B optimized (all improvements) ===================="
  date
  WHEAT_SEG_TIMING=1 python src/run_reconstruction.py \
    plot=${PLOT} experiment_name=${EXP} ${PP} \
    run_seg=true run_eval=false run_eval_2d=false \
    segmentation_3d.use_mask_cache=true \
    segmentation_3d.frustum_cull=true \
    segmentation_3d.exp_name=seg_opt

  # ---- 4) exactness gate: optimized MUST equal baseline (md5 + 2DSeg) ----
  echo "==================== ${PLOT}  GATE (compare_seg_runs.py) ===================="
  SEG=results/reconstruction/fip/${PLOT}/vanilla_3dgs/${EXP}/segmentation_3d
  python src/analysis/compare_seg_runs.py "${SEG}/seg_opt" "${SEG}/seg_baseline"
  RC=$?
  BASE_MD5=$(md5sum "${SEG}/seg_baseline/all_obj_labels.pth" 2>/dev/null | awk '{print $1}')
  OPT_MD5=$(md5sum  "${SEG}/seg_opt/all_obj_labels.pth"      2>/dev/null | awk '{print $1}')
  if [ "$RC" = "0" ]; then V="PASS ✅"; else V="FAIL ❌"; fi
  printf "%-10s  baseline=%s  optimized=%s  -> %s\n" "$PLOT" "${BASE_MD5:-MISSING}" "${OPT_MD5:-MISSING}" "$V" >> "$SUMMARY"
  echo "  ${PLOT}: ${V}"
done

# ---- final summary table for this job's plots ----
echo ""
echo "################################################################"
echo "#  FIP SEG-OPTIMIZATION LOSSLESSNESS — SUMMARY (this job)"
echo "#  PASS = optimized seg is byte-identical (md5 + 2DSeg) to the baseline"
echo "################################################################"
cat "$SUMMARY"
