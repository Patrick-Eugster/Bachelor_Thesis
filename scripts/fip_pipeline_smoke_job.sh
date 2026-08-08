#!/bin/bash -l
# FIP FULL-PIPELINE SMOKE TEST on ONE plot — verifies the whole chain still works end to end,
# especially 3D segmentation + render_360 (the steps we haven't exercised in a while).
# Chain: train -> render -> metrics -> seg -> export colored PLY -> render_360 (the video!) -> eval -> eval_2d.
# Engine: gsplat (default) + use_principal_point=true (the validated FIP setup). 15k iters for speed.
#
# PREREQ: the render_360 mp4 stitch needs libopenh264.so.5 in the wheat3dgs env (Cisco openh264 2.0.0).
#         Do the one-time login-node fix FIRST (see chat) or the video step will fail at mp4 stitch.
#
# SUBMIT (full absolute path required):
#   sbatch /cluster/project/cropsci/peugste/wheat3dgs/scripts/fip_pipeline_smoke_job.sh
#
#SBATCH --job-name=fip_pipe_smoke
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=6
#SBATCH --time=03:59:00
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_pipe_smoke_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_pipe_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=peugste@ethz.ch

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

REPO=/cluster/project/cropsci/peugste/wheat3dgs
cd "$REPO"
nvidia-smi
which ffmpeg && ffmpeg -version 2>&1 | head -1   # sanity: ffmpeg loads (openh264 fix applied)

PLOT=plot_461
EXP=pipeline_smoke_461
export WHEAT_RUN_REPORT=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/run_report_${SLURM_JOB_ID}.txt

echo ""
echo "========================================"
echo "  $PLOT  FULL pipeline smoke (gsplat + pp, 15k) incl render_360"
echo "========================================"
python src/run_reconstruction.py \
  plot=$PLOT \
  run_train=true run_render=true run_metrics=true \
  run_seg=true run_render_360=true run_eval=true run_eval_2d=true \
  reconstruction.iterations=15000 \
  reconstruction.use_principal_point=true \
  segmentation_3d.detection_method=yolo_sam_v1 \
  segmentation_3d.mask_gen_experiment=initial \
  allow_overwrite=true \
  experiment_name=$EXP

echo ""
echo "========================================"
echo "  DONE. Key outputs to check:"
echo "   video : results/reconstruction/fip/$PLOT/vanilla_3dgs/$EXP/segmentation_3d/*/3DSeg/  (mp4)"
echo "   report: results/reconstruction/fip/$PLOT/vanilla_3dgs/$EXP/run_report.txt"
echo "========================================"
cat "$REPO/results/reconstruction/fip/$PLOT/vanilla_3dgs/$EXP/run_report.txt" 2>/dev/null || true
