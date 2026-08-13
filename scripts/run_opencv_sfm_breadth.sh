#!/bin/bash
# LOCAL opencv-SfM generation for the OTHER 4 canonical sessions (prerequisite for phone_recon_opencv_breadth_job.sh).
# The camera-model A4 runs only produced opencv/ SfM for fA/0618, fA/0715, fD/0627, fD/0706. This builds it for
# the remaining 4 (fA/0627, fA/0706, fD/0618, fD/0715) so the opencv 3DGS breadth arm has inputs to train on.
#
# Runs COLMAP with camera=OPENCV (fit distortion -> image_undistorter warps it out -> cropped PINHOLE frame),
# ALIKED+LightGlue + exhaustive + single_camera (baseline settings), isolated in <session>/opencv/. Also runs
# compare_to_agisoft (writes only *_opencv-suffixed files). WRITE-SAFE: only touches <session>/opencv/ + the
# _opencv compare files; baseline sparse/ and all other arms untouched. ~10-12 min/session locally (GPU ALIKED).
#
# AFTER this finishes, rsync the 4 opencv/ folders up to Euler, then submit phone_recon_opencv_breadth_job.sh.
cd /workspace
OUT=/tmp/claude-0/-workspace/e6a109d2-41f2-481c-a1e9-2e00efdd4d44/scratchpad/opencv_sfm_breadth_results.txt
: > "$OUT"
printf "%-18s %-8s %8s %6s %12s %11s %9s\n" session model reg sub trans_med_mm rot_med_deg colmap_s >> "$OUT"

CAM=OPENCV
VD=opencv

for FP in field_A/20250627 field_A/20250706 field_D/20250618 field_D/20250715; do
  F=${FP%/*}; P=${FP#*/}
  IN="input_plots/phone/$FP/$VD/sparse/0"
  if [ -d "$IN" ] && [ -n "$(ls -A "$IN" 2>/dev/null)" ]; then
    echo "SKIP $FP : opencv/ already exists ($IN)"; continue
  fi
  echo "==================== $FP : $CAM -> $VD/ ===================="
  t0=$SECONDS
  python src/preprocessing/run_colmap.py field=$F plot=$P front_end=aliked camera=$CAM variant_dir=$VD 2>&1 | tail -5
  t_colmap=$(( SECONDS - t0 ))
  echo "==================== $FP : $CAM compare  (colmap ${t_colmap}s) ===================="
  python src/preprocessing/compare_to_agisoft.py field=$F plot=$P \
      ours_sparse_dir=$VD/sparse/0 output_file=logs/compare_to_agisoft_$VD.json 2>&1 | tail -5

  python3 - "$FP" "$CAM" "$VD" "$OUT" "$t_colmap" <<'PY'
import json,os,sys,glob
fp,cam,vd,out,tc=sys.argv[1:6]
b=f"input_plots/phone/{fp}"
reg="?"; sub="?"; tr="?"; ro="?"
try:
    s=json.load(open(f"{b}/{vd}/logs/colmap_summary.json")); reg=f"{s.get('registered','?')}/{s.get('input_images','?')}"
except Exception: pass
try: sub=str(len([d for d in glob.glob(f"{b}/{vd}/distorted/sparse/*") if os.path.isdir(d)]))
except Exception: pass
try:
    c=json.load(open(f"{b}/logs/compare_to_agisoft_{vd}.json"))
    tr=f"{c['translation_error_m']['median_m']*1000:.1f}"; ro=f"{c['rotation_error_deg']['median_deg']:.2f}"
except Exception: pass
open(out,"a").write(f"{fp:<18} {cam:<8} {reg:>8} {sub:>6} {tr:>12} {ro:>11} {tc:>9}\n")
PY
done
echo "ALL DONE" >> "$OUT"
echo "ALL DONE — now rsync the 4 opencv/ folders to Euler, then submit phone_recon_opencv_breadth_job.sh"
cat "$OUT"
