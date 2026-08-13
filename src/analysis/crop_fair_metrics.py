"""Fast crop-fair recon metrics: whole vs inner vs ROI PSNR/SSIM per phone arm, WITHOUT LPIPS (so no
VGG memory blowup — runs locally in ~1-2 min). Answers whether opencv's whole-image PSNR edge over
pinhole survives a fair region. LPIPS-inner can be added later on Euler via the full metrics.py."""
import glob, os, sys, numpy as np, torch
from PIL import Image
import torchvision.transforms.functional as tf
sys.path.insert(0, "src")
from reconstruction.metrics import build_marker_ctx, project_markers, build_test_names, inner_box, _ssim_map, laplacian_var
from gaussians.utils.image_utils import psnr
from gaussians.utils.loss_utils import ssim

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sharp_ratio(r, g):
    """Render/GT sharpness ratio (Laplacian variance) — cheap, no VGG. >1 = render sharper than GT."""
    sg = laplacian_var(g)
    return laplacian_var(r) / sg if sg > 0 else 0.0


def masked(r, g, mask, box):
    """PSNR + SSIM over the polygon MASK, + sharpness ratio over the bbox crop (same split as
    metrics.roi_metrics: PSNR/SSIM polygon-masked, sharpness on the rectangle). Minus LPIPS."""
    m = torch.from_numpy(np.ascontiguousarray(mask)).to(r.device).bool()
    se = ((r - g) ** 2).mean(dim=-3).squeeze(0)
    p = float(-10 * torch.log10(se[m].mean()))
    s = float(_ssim_map(r, g).mean(dim=-3).squeeze(0)[m].mean())
    x0, y0, x1, y1 = box
    sh = sharp_ratio(r[..., y0:y1, x0:x1].contiguous(), g[..., y0:y1, x0:x1].contiguous())
    return p, s, sh


def arm_metrics(model, src):
    """Return dict with whole/inner/roi (PSNR,SSIM) averaged over the arm's test views."""
    rdir = sorted(glob.glob(model + "/test/*/renders"))
    if not rdir:
        return None
    rdir = rdir[0]
    ctx = build_marker_ctx(src)
    test_names = build_test_names(src) if ctx is not None else None
    W = {"whole": [], "inner": [], "roi": []}
    for f in sorted(os.listdir(rdir)):
        r = tf.to_tensor(Image.open(os.path.join(rdir, f)))[:3].unsqueeze(0).to(DEV)
        g = tf.to_tensor(Image.open(os.path.join(rdir.replace("/renders", "/gt"), f)))[:3].unsqueeze(0).to(DEV)
        H, Wd = r.shape[-2:]
        W["whole"].append((psnr(r, g).mean().item(), ssim(r, g).item(), sharp_ratio(r, g)))
        x0, y0, x1, y1 = inner_box(H, Wd)
        ri, gi = r[..., y0:y1, x0:x1].contiguous(), g[..., y0:y1, x0:x1].contiguous()
        W["inner"].append((psnr(ri, gi).mean().item(), ssim(ri, gi).item(), sharp_ratio(ri, gi)))
        if ctx is not None:
            k = int(os.path.splitext(f)[0]); stem = test_names[k] if k < len(test_names) else None
            box, _, mask = project_markers(ctx, stem, H, Wd) if stem else (None, None, None)
            if box is not None:
                W["roi"].append(masked(r, g, mask, box))
        del r, g
    out = {}
    for kk, v in W.items():
        if v:
            a = np.array(v); out[kk] = (a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), len(v))  # psnr, ssim, sharp, n
    return out


def main():
    arms = sorted(glob.glob("results/reconstruction/phone/**/baseline", recursive=True))
    rows = {}
    for model in arms:
        if not os.path.isdir(model + "/test"):
            continue
        parts = model.split("/"); f, d = parts[3], parts[4]; vi = parts.index("vanilla_3dgs")
        variant = parts[vi - 1] if parts[vi - 1] != d else "pinhole"
        src = os.path.join("input_plots/phone", f, d, "" if variant == "pinhole" else variant)
        m = arm_metrics(model, src)
        if m:
            rows.setdefault(f"{f.split('_')[1]}/{d[4:]}", {})[variant] = m
            print(f"done {f}/{d} {variant}")

    order = ["pinhole", "opencv", "radial", "agisoft", "agisoft_2group_old"]
    print("\n=== WHOLE vs INNER PSNR (dB) per arm  [inner = centered 80%, crop-fair] ===")
    print(f"{'session':8} " + " ".join(f"{a[:8]:>17}" for a in order))
    for s in sorted(rows):
        cells = []
        for a in order:
            if a in rows[s]:
                w = rows[s][a].get("whole"); i = rows[s][a].get("inner")
                cells.append(f"{w[0]:5.2f}/{i[0]:5.2f}" if w and i else f"{'--':>11}")
            else:
                cells.append(f"{'--':>11}")
        print(f"{s:8} " + " ".join(f"{c:>17}" for c in cells))
    print("cell = WHOLE/INNER PSNR")

    print("\n=== pinhole vs opencv: does opencv's edge survive the inner crop? ===")
    print(f"{'session':8} {'whole Δ':>9} {'inner Δ':>9}   (Δ = opencv - pinhole, dB)")
    for s in sorted(rows):
        if "pinhole" in rows[s] and "opencv" in rows[s]:
            wd = rows[s]["opencv"]["whole"][0] - rows[s]["pinhole"]["whole"][0]
            idd = rows[s]["opencv"]["inner"][0] - rows[s]["pinhole"]["inner"][0]
            print(f"{s:8} {wd:>+9.3f} {idd:>+9.3f}")

    print("\n=== SHARPNESS ratio (render/GT, >1 = render sharper) — whole / inner per arm ===")
    print(f"{'session':8} " + " ".join(f"{a[:8]:>13}" for a in order))
    for s in sorted(rows):
        cells = []
        for a in order:
            if a in rows[s]:
                w = rows[s][a].get("whole"); i = rows[s][a].get("inner")
                cells.append(f"{w[2]:.2f}/{i[2]:.2f}" if w and i else f"{'--':>9}")
            else:
                cells.append(f"{'--':>9}")
        print(f"{s:8} " + " ".join(f"{c:>13}" for c in cells))
    print("cell = WHOLE/INNER sharpness ratio")

    print("\n=== ROI (plot region) where markers exist — PSNR + sharpness ===")
    for s in sorted(rows):
        for a in order:
            if a in rows[s] and "roi" in rows[s][a]:
                roi = rows[s][a]["roi"]; wh = rows[s][a]["whole"]
                print(f"  {s:8} {a:10} whole PSNR {wh[0]:5.2f} sharp {wh[2]:.2f}  |  ROI PSNR {roi[0]:5.2f} SSIM {roi[1]:.3f} sharp {roi[2]:.2f}  ({roi[3]} views)")


if __name__ == "__main__":
    main()
