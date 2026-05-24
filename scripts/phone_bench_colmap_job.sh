#!/bin/bash -l
#SBATCH -J phone_bench_colmap
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=8G      # 48 GB total (6 × 8G) — phone ~93 full-res images in CPU RAM
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00       # ~2-3h per session × 4 = ~10h, 12h gives buffer
#SBATCH --output=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_bench_colmap_%j.out
#SBATCH --error=/cluster/project/cropsci/peugste/wheat3dgs/slurm_logs/phone_bench_colmap_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=peugste@ethz.ch

module purge
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs
which python
python --version
conda info --envs
module load stack/2025-06 gcc/12.2.0 cuda/12.6.2 eth_proxy

nvidia-smi

cd /cluster/project/cropsci/peugste/wheat3dgs

for args in "field_A 20250609" "field_A 20250618" "field_D 20250523" "field_D 20250530"; do
  field=$(echo $args | cut -d' ' -f1)
  date=$(echo $args | cut -d' ' -f2)
  echo ""
  echo "========================================"
  echo "  $field / $date  (COLMAP)"
  echo "========================================"
  python src/run_reconstruction.py \
    dataset=phone plot=$field date=$date \
    run_train=true run_render=true run_metrics=true \
    experiment_name=colmap_bench
done