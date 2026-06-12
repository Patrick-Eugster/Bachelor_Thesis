import os
import datetime


def resolve_experiment_name(cfg):
    """Return final experiment name based on experiment_name and prepend_date.

    ""        → pure timestamp "2025-04-28_1430"            (prepend_date ignored)
    "initial" → "initial"                                   (prepend_date ignored)
    "my_run"  + prepend_date=False → "my_run"
    "my_run"  + prepend_date=True  → "2025-04-28_my_run"
    """
    if not cfg.experiment_name:
        # pure timestamp — prepend_date ignored to avoid double date
        return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    if cfg.experiment_name == "initial":
        # fixed scratch name — never prepend date
        return "initial"
    if cfg.prepend_date:
        return f"{datetime.datetime.now().strftime('%Y-%m-%d')}_{cfg.experiment_name}"
    return cfg.experiment_name


def _plot_subpath(cfg):
    """Return the sub-path under input_dir / result_dir for this dataset+plot.

    FIP (date=""): "plot_461"
    Phone (date="20250618"): "field_A/20250618"
    """
    date = str(getattr(cfg, 'date', ''))
    return os.path.join(cfg.plot, date) if date else cfg.plot


def get_dataset_path(cfg):
    """Absolute path to the input plot folder.

    FIP:                input_plots/fip/plot_461/
    Phone (COLMAP):     input_plots/phone/field_A/20250618/
    Phone (Agisoft):    input_plots/phone/field_A/20250618/agisoft/   (use_agisoft_sfm=true)
    """
    base = os.path.join(cfg.dataset.input_dir, _plot_subpath(cfg))
    if getattr(cfg, 'use_agisoft_sfm', False):
        return os.path.join(base, 'agisoft')
    return base


def get_mask_generation_result_path(cfg, plot_name):
    """Derive output path for a given plot name. The method-name subfolder comes from
    cfg.method.name (Option C) so each detection method gets its own result tree; defaults
    to "yolo_sam_v1" when no method is set, so existing paths are unchanged.

    yolo_sam_v1:   results/mask_generation/fip/plot_461/yolo_sam_v1/{experiment}/
    sahi_yolo_sam: results/mask_generation/fip/plot_461/sahi_yolo_sam/{experiment}/
    """
    exp_name = resolve_experiment_name(cfg)
    method_name = cfg.method.name if (hasattr(cfg, "method") and hasattr(cfg.method, "name")) else "yolo_sam_v1"
    return os.path.join(cfg.dataset.result_dir_masks, plot_name, method_name, exp_name)


def get_reconstruction_model_path(cfg, exp_name):
    """Derive 3DGS model output path.

    FIP:                results/reconstruction/fip/plot_461/vanilla_3dgs/{experiment}/
    Phone (COLMAP):     results/reconstruction/phone/field_A/20250618/vanilla_3dgs/{experiment}/
    Phone (Agisoft):    results/reconstruction/phone/field_A/20250618/agisoft/vanilla_3dgs/{experiment}/
                        — mirrors input layout so an Agisoft run never overwrites the COLMAP run
                          even when both share experiment_name="initial"
    """
    base = os.path.join(cfg.dataset.result_dir_recon, _plot_subpath(cfg))
    if getattr(cfg, 'use_agisoft_sfm', False):
        base = os.path.join(base, 'agisoft')
    return os.path.join(base, "vanilla_3dgs", exp_name)


def get_seg_source_dir(cfg):
    """Derive path to detection results used as input for reconstruction. The method-name
    subfolder comes from cfg.segmentation_3d.detection_method (Option C, default "yolo_sam_v1"),
    so segmentation can read a SAHI run's masks by setting detection_method=sahi_yolo_sam.

    yolo_sam_v1:   results/mask_generation/fip/plot_461/yolo_sam_v1/{mask_gen_experiment}/
    sahi_yolo_sam: results/mask_generation/fip/plot_461/sahi_yolo_sam/{mask_gen_experiment}/
    """
    method_dir = cfg.segmentation_3d.get("detection_method", "yolo_sam_v1")
    base = os.path.join(cfg.dataset.result_dir_masks, _plot_subpath(cfg), method_dir, cfg.segmentation_3d.mask_gen_experiment)
    return os.path.join(base, "yolosam") if cfg.use_yolosam_source else base


def get_log_file(model_path, exp_name):
    """Derive log file path inside the model folder."""
    return os.path.join(model_path, "seg_logs", f"{exp_name}.txt")
