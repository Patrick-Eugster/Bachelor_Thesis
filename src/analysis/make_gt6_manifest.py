"""Build configs/manifests/gt6.json — the 6 hand-labelled GT images (image + GT instance mask + YOLO
low-floor bboxes_with_conf) that the conf sweep (sweep_conf_mask_ap.py) scores. Repo-relative paths, so
the same manifest works locally and on Euler. Re-run whenever the GT set changes."""
import json
import os

STEMS = {  # noqa: E305
    ("field_A", "20250627"): "IMG_20250627_101017",
    ("field_A", "20250706"): "IMG_20250706_121516",
    ("field_A", "20250715"): "IMG_20250715_153912",
    ("field_D", "20250627"): "IMG_20250627_102602",
    ("field_D", "20250706"): "IMG_20250706_123031",
    ("field_D", "20250715"): "IMG_20250715_155705",
}


def _active_instances(sets_dir):
    """Resolve the ACTIVE instance mask via manifest.json (active set name -> its 'file' stem), so a future
    re-label that makes another set active never silently scores against the stale set0. Falls back to
    set0_instances.png only if there is no manifest.json."""
    man_path = os.path.join(sets_dir, "manifest.json")
    if os.path.exists(man_path):
        m = json.load(open(man_path))
        active = m.get("active")
        file_stem = next((s["file"] for s in m.get("sets", []) if s.get("name") == active), "set0")
        return os.path.join(sets_dir, f"{file_stem}_instances.png")
    return os.path.join(sets_dir, "set0_instances.png")


def main():
    """Write the manifest, refusing if any of the 18 files is missing."""
    man, allok = [], True
    for (f, d), s in STEMS.items():
        img = f"input_plots/phone/{f}/{d}/images/{s}.jpg"
        gt = _active_instances(f"input_plots/phone/{f}/{d}/manual_label/{s}_sets")
        bb = f"results/mask_generation/phone/{f}/{d}/yolo_sam_v1/metrics_v1/bboxes_with_conf/{s}.pt"
        ok = all(os.path.exists(p) for p in (img, gt, bb))
        allok &= ok
        print(("OK  " if ok else "MISS") + f" {f}/{d} {s}  gt={os.path.basename(gt)}")
        man.append({"stem": s, "image": img, "gt": gt, "bbox": bb})
    if not allok:
        raise SystemExit("some GT files missing — manifest NOT written")
    os.makedirs("configs/manifests", exist_ok=True)
    json.dump(man, open("configs/manifests/gt6.json", "w"), indent=2)
    print("wrote configs/manifests/gt6.json")


if __name__ == "__main__":
    main()
