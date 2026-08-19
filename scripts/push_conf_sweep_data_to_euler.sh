#!/usr/bin/env bash
# Push the GT data the conf sweep (scripts/conf_sweep_maskap_job.sh) needs to Euler — the 18 files named by
# configs/manifests/gt6.json: for each of the 6 GT images its images/ jpg, its manual_label/*_sets/
# set0_instances.png, and its results/.../metrics_v1/bboxes_with_conf/*.pt. ~30MB total. These live under
# input_plots/ and results/ which are gitignored, so a normal code push does NOT carry them — this does.
#
# The manifest itself (configs/manifests/gt6.json) is git-tracked, so it arrives with a normal code rsync.
#
# Run FROM THE HOST REPO ROOT. rsync --files-from recreates the exact relative paths under the Euler repo;
# nothing else is transferred, nothing on Euler is overwritten except these 18 files.
set -eu
cd /home/patrick/Bachelor_Thesis
DEST=peugste@euler.ethz.ch:/cluster/project/cropsci/peugste/wheat3dgs/

test -f configs/manifests/gt6.json || { echo "missing configs/manifests/gt6.json (run make_gt6_manifest.py)"; exit 1; }

LIST=$(mktemp)
python - <<'PY' > "$LIST"
import json
for x in json.load(open("configs/manifests/gt6.json")):
    print(x["image"]); print(x["gt"]); print(x["bbox"])
PY

echo "=== pushing $(wc -l < "$LIST") GT-data files (~30MB) to Euler ==="
rsync -avz --files-from="$LIST" ./ "$DEST"
rm -f "$LIST"

echo ""
echo "=== SAM weights check — the job needs these on Euler (NOT pushed here, 2.5GB+): ==="
echo "  ssh peugste@euler.ethz.ch 'ls -la /cluster/project/cropsci/peugste/wheat3dgs/src/mask_generation/weights/{sam_vit_h_4b8939.pth,sam2.1_l.pt}'"
echo "If either is missing, push it once (big, one-time):"
echo "  rsync -avz src/mask_generation/weights/sam_vit_h_4b8939.pth src/mask_generation/weights/sam2.1_l.pt $DEST""src/mask_generation/weights/"
