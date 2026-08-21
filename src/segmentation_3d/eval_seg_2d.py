import cv2
import glob
import json
import os
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser
from gaussians.arguments import ModelParams, PipelineParams, get_combined_args


def compute_metrics(binary_gt, binary_pred):
    """Compute pixel-level segmentation metrics between two boolean numpy arrays."""
    intersection = np.logical_and(binary_gt, binary_pred).sum()
    union        = np.logical_or(binary_gt, binary_pred).sum()
    tp = int(intersection)
    fp = int(binary_pred.sum() - intersection)
    fn = int(binary_gt.sum() - intersection)
    tn = int(np.logical_and(~binary_gt, ~binary_pred).sum())

    iou       = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MSE: fraction of pixels where GT and pred disagree (both are 0/1 so squares are 0 or 1)
    gt_f   = binary_gt.astype(np.float32)
    pred_f = binary_pred.astype(np.float32)
    mse      = float(np.mean((gt_f - pred_f) ** 2))
    ssim_val = float(ssim(gt_f, pred_f, data_range=1.0))

    # MCC: robust to class imbalance — uses all four quadrants including TN
    mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    # Balanced accuracy: average of recall (TPR) and specificity (TNR) — robust for imbalanced data
    specificity      = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc     = (recall + specificity) / 2.0

    # FPR: fraction of background pixels wrongly predicted as wheat
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "iou": float(iou), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "mse": mse, "ssim": ssim_val,
        "mcc": float(mcc), "balanced_acc": float(balanced_acc), "fpr": float(fpr),
    }


def count_gt_heads(manual_label_dir, stem):
    """Count GT wheat heads by counting lines in the YOLO .txt annotation file."""
    txt_path = os.path.join(manual_label_dir, f"{stem}.txt")
    if not os.path.exists(txt_path):
        return None
    with open(txt_path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    return len(lines)


def count_pred_heads(model_path, exp_name, stem):
    """Count predicted wheat heads as distinct non-zero IDs in the 2DSeg label map for this camera."""
    pt_path = os.path.join(model_path, "segmentation_3d", exp_name, "2DSeg", f"{stem}.pt")
    if not os.path.exists(pt_path):
        return None
    label_map = torch.load(pt_path)
    unique_ids = torch.unique(label_map)
    # exclude 0 (background)
    return int((unique_ids != 0).sum().item())


def make_visualization(binary_gt, binary_pred, base_img, alpha=0.6):
    """Color-coded overlay: green=TP, blue=FN, red=FP, blended over the RGB image (BGR colors)."""
    h, w = binary_gt.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    # old scheme (BGR): TP=[0,0,255] red, FN=[128,128,128] gray, FP=[128,213,255] light yellow.
    # switched to green=correct / red=FP / blue=FN because red-for-TP read as an error at a glance.
    vis[np.logical_and(binary_gt,  binary_pred)]  = [0,   255, 0]    # green: TP (correct)
    vis[np.logical_and(binary_gt,  ~binary_pred)] = [255, 0,   0]    # blue:  FN (missed head)
    vis[np.logical_and(~binary_gt, binary_pred)]  = [0,   0,   255]  # red:   FP (invented head)
    base_f = base_img.astype(np.float32) / 255.0
    vis_f  = vis.astype(np.float32) / 255.0
    blended = cv2.addWeighted(vis_f, alpha, base_f, 1 - alpha, 0)
    return (blended * 255).astype(np.uint8)


def eval_seg_2d(source_path, model_path, exp_name, pred_type="wheat", threshold=128):
    """
    For each *_gt_mask.png in manual_label/, find the matching pipeline prediction in
    test/ or train/ segmentation output, compute pixel-level metrics and save a
    color-coded visualization. Returns a list of per-camera metric dicts.

    pred_type: "wheat" = Wheat3DGS pipeline (default)
               "fruit" = FruitNeRF baseline (see commented block below — not wired up yet)
    """
    manual_label_dir = os.path.join(source_path, "manual_label")
    if not os.path.isdir(manual_label_dir):
        print(f"  No manual_label/ at {manual_label_dir} — skipping")
        return []

    gt_mask_files = sorted(glob.glob(os.path.join(manual_label_dir, "*_gt_mask.png")))
    if not gt_mask_files:
        print(f"  No *_gt_mask.png found in {manual_label_dir} — skipping")
        return []

    out_dir = os.path.join(model_path, "segmentation_3d", exp_name, "eval_2d")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for gt_mask_path in gt_mask_files:
        # stem = camera name, e.g. FPWW036_SR0461_FIP2_cam_12
        stem = os.path.basename(gt_mask_path).replace("_gt_mask.png", "")

        gt_gray  = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_gray is None:
            print(f"  Could not read GT mask for {stem}")
            continue

        pred_gray = None  # assigned inside the pred_type branches below
        if pred_type == "wheat":
            # find matching prediction — eval_wheatgs writes to test/ and optionally train/
            pred_path = None
            for split in ("test", "train"):
                candidate = os.path.join(model_path, split, "segmentation", f"{stem}.png")
                if os.path.exists(candidate):
                    pred_path = candidate
                    break
            if pred_path is None:
                print(f"  No prediction for {stem} — run eval step (run_eval) first")
                continue
            pred_gray = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # --- FruitNeRF baseline comparison (not wired up yet) ---
        # To compare against FruitNeRF: produce a *_fruitnerf.png alongside *_gt_mask.png
        # in manual_label/, then switch pred_type="fruit" when calling eval_seg_2d.
        # elif pred_type == "fruit":
        #     fruit_candidates = glob.glob(os.path.join(manual_label_dir, f"{stem}_fruitnerf.png"))
        #     if not fruit_candidates:
        #         print(f"  No FruitNeRF prediction for {stem}")
        #         continue
        #     pred_gray = cv2.imread(fruit_candidates[0], cv2.IMREAD_GRAYSCALE)
        #     assert gt_gray.shape == pred_gray.shape, f"Shape mismatch: {gt_gray.shape} vs {pred_gray.shape}"

        if pred_gray is None:
            print(f"  Could not read prediction image for {stem}")
            continue

        # prediction resolution may differ from GT — resize to match
        if gt_gray.shape != pred_gray.shape:
            pred_gray = cv2.resize(pred_gray, (gt_gray.shape[1], gt_gray.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        binary_gt   = gt_gray   >= threshold
        binary_pred = pred_gray >= threshold

        m = compute_metrics(binary_gt, binary_pred)
        m["gt_head_count"]   = count_gt_heads(manual_label_dir, stem)
        m["pred_head_count"] = count_pred_heads(model_path, exp_name, stem)
        # count error ratio: (pred - GT) / GT — negative = under-count, positive = over-count
        if m["gt_head_count"] and m["pred_head_count"] is not None:
            m["count_error_ratio"] = (m["pred_head_count"] - m["gt_head_count"]) / m["gt_head_count"]
        else:
            m["count_error_ratio"] = None
        m["camera"] = stem
        results.append(m)

        # visualization — blend color-coded TP/FP/FN over the raw RGB image
        img_candidates = glob.glob(os.path.join(source_path, "images", f"{stem}.*"))
        if img_candidates:
            base_img = cv2.imread(img_candidates[0])
            if base_img is not None:
                if base_img.shape[:2] != binary_gt.shape:
                    base_img = cv2.resize(base_img, (binary_gt.shape[1], binary_gt.shape[0]))
                vis = make_visualization(binary_gt, binary_pred, base_img)
                cv2.imwrite(os.path.join(out_dir, f"{stem}_eval2d.png"), vis)

    return results


def _print_table(results):
    """Print results as a two-line-per-camera ASCII table with a mean row at the bottom.
    Line 1: pixel metrics — IoU, Prec, Rec, F1, MSE, SSIM
    Line 2: imbalance-robust + count metrics — MCC, BalAcc, FPR, heads pred/GT, count error ratio
    """
    col_cam = max(len(r["camera"]) for r in results)
    pad     = " " * col_cam

    header1 = f"{'Camera':<{col_cam}}  {'IoU':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'MSE':>6}  {'SSIM':>6}"
    header2 = f"{pad}  {'MCC':>6}  {'BalAcc':>6}  {'FPR':>6}  {'Heads p/GT':>11}  {'CntErr%':>8}"
    sep     = "-" * max(len(header1), len(header2))

    print(f"\n{header1}")
    print(f"{header2}")
    print(sep)

    for r in results:
        if r["gt_head_count"] is not None and r["pred_head_count"] is not None:
            heads    = f"{r['pred_head_count']}/{r['gt_head_count']}"
            cnt_err  = f"{r['count_error_ratio']*100:>+.1f}%" if r["count_error_ratio"] is not None else "n/a"
        else:
            heads   = "n/a"
            cnt_err = "n/a"
        print(f"{r['camera']:<{col_cam}}  {r['iou']:>6.4f}  {r['precision']:>6.4f}  "
              f"{r['recall']:>6.4f}  {r['f1']:>6.4f}  {r['mse']:>6.4f}  {r['ssim']:>6.4f}")
        print(f"{pad}  {r['mcc']:>6.4f}  {r['balanced_acc']:>6.4f}  {r['fpr']:>6.4f}  "
              f"{heads:>11}  {cnt_err:>8}")

    print(sep)

    # mean row
    avg_keys = ("iou", "precision", "recall", "f1", "mse", "ssim", "mcc", "balanced_acc", "fpr")
    means = {k: np.mean([r[k] for r in results]) for k in avg_keys}
    counts = [(r["gt_head_count"], r["pred_head_count"]) for r in results
              if r["gt_head_count"] is not None and r["pred_head_count"] is not None]
    if counts:
        gt_total   = sum(g for g, _ in counts)
        pred_total = sum(p for _, p in counts)
        heads_summary  = f"{pred_total}/{gt_total}"
        cerr_vals = [r["count_error_ratio"] for r in results if r["count_error_ratio"] is not None]
        cerr_summary = f"{np.mean(cerr_vals)*100:>+.1f}%" if cerr_vals else "n/a"
    else:
        heads_summary = "n/a"
        cerr_summary  = "n/a"

    print(f"{'Mean':<{col_cam}}  {means['iou']:>6.4f}  {means['precision']:>6.4f}  "
          f"{means['recall']:>6.4f}  {means['f1']:>6.4f}  {means['mse']:>6.4f}  {means['ssim']:>6.4f}")
    print(f"{pad}  {means['mcc']:>6.4f}  {means['balanced_acc']:>6.4f}  {means['fpr']:>6.4f}  "
          f"{heads_summary:>11}  {cerr_summary:>8}")
    print()


if __name__ == "__main__":
    parser = ArgumentParser(description="Pixel-level 2D segmentation evaluation vs manual GT masks")
    model    = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--exp_name", type=str, required=True, help="Segmentation experiment name")
    args    = get_combined_args(parser)
    dataset = model.extract(args)

    results = eval_seg_2d(dataset.source_path, dataset.model_path, args.exp_name)

    if not results:
        print("No results computed.")
    else:
        out_dir  = os.path.join(dataset.model_path, "segmentation_3d", args.exp_name, "eval_2d")
        out_path = os.path.join(out_dir, "metrics_2d.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out_path}")
        _print_table(results)
