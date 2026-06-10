#!/bin/bash -l
#SBATCH -J fip_eval_iters
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00        # eval only (no training): render 15k+30k test views + metrics, 7 plots × 2 arms.
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_eval_iters_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/fip_eval_iters_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

# ── 15k-vs-30k iteration comparison (NO retraining) ──
# A 30k run already saved point_cloud/iteration_15000 (save_iterations default), and because the
# LR schedule (position_lr_max_steps=30000) + densification schedule (densify_until_iter=11000) are
# anchored to fixed step counts, iter-15000 of a 30k run == a standalone 15k run in this codebase.
# So we just render the 15k AND 30k test views into the same test/ folder and run metrics once —
# metrics.py iterates every render subfolder, so results.json ends up with BOTH ours_15000 and
# ours_30000 (PSNR/SSIM/LPIPS each). Compare to decide if 15k is "good enough" (saves ~half the time)
# or 30k earns its cost. Done for both arms: vanilla gsplat (test_gsplat_full) + AbsGrad (test_absgrad).
# render.py uses get_combined_args → it auto-loads use_principal_point / resolution from each model's
# saved cfg_args, so no per-flag plumbing is needed here.

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi
cd /cluster/project/cropsci/peugste/wheat3dgs

for EXP in test_gsplat_full test_absgrad; do
  for PLOT in plot_461 plot_462 plot_463 plot_464 plot_465 plot_466 plot_467; do
    SRC=input_plots/fip/${PLOT}
    MODEL=results/reconstruction/fip/${PLOT}/vanilla_3dgs/${EXP}

    # skip arms/plots that weren't trained (e.g. test_absgrad before its job has run)
    if [ ! -d "${MODEL}/point_cloud/iteration_15000" ] || [ ! -d "${MODEL}/point_cloud/iteration_30000" ]; then
      echo ">>> SKIP ${EXP}/${PLOT}: missing iteration_15000 or iteration_30000 checkpoint"
      continue
    fi

    echo ""
    echo "========================================"
    echo "  ${EXP} / ${PLOT}  — render 15k + 30k test views, then metrics"
    echo "========================================"
    # --skip_train: metrics only reads test/, so we don't need train renders. Both land in test/.
    python src/reconstruction/render.py -s ${SRC} -m ${MODEL} --iteration 15000 --skip_train
    python src/reconstruction/render.py -s ${SRC} -m ${MODEL} --iteration 30000 --skip_train
    python src/reconstruction/metrics.py -m ${MODEL}

    echo "----- results.json (${EXP}/${PLOT}) — has ours_15000 AND ours_30000 -----"
    cat ${MODEL}/results.json
    echo ""
  done
done

echo ""
echo "========================================"
echo "  DONE — each results.json now contains ours_15000 + ours_30000"
echo "  Compare LPIPS + (later) seg IoU between the two iterations to decide 15k vs 30k."
echo "========================================"
