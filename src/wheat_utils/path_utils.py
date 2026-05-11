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


def get_dataset_path(cfg):
    """Absolute path to the input plot: input_plots/fip/plot_461/ — reconstruction only."""
    return os.path.join(cfg.dataset.input_dir, cfg.plot)


def get_mask_generation_result_path(cfg, plot_name):
    """Derive output path for a given plot name.

    results/mask_generation/fip/plot_461/yolo_sam_v1/{experiment}/
    """
    exp_name = resolve_experiment_name(cfg)
    return os.path.join(cfg.dataset.result_dir_masks, plot_name, "yolo_sam_v1", exp_name)


def get_reconstruction_model_path(cfg, exp_name):
    """Derive 3DGS model output path.

    results/reconstruction/fip/plot_461/vanilla_3dgs/{experiment}/
    """
    return os.path.join(cfg.dataset.result_dir_recon, cfg.plot, "vanilla_3dgs", exp_name)


def get_seg_source_dir(cfg):
    """Derive path to detection results used as input for reconstruction.

    results/mask_generation/fip/plot_461/yolo_sam_v1/{detection_experiment}/
    """
    base = os.path.join(cfg.dataset.result_dir_masks, cfg.plot, "yolo_sam_v1", cfg.detection_experiment)
    return os.path.join(base, "yolosam") if cfg.use_yolosam_source else base


def get_log_file(model_path, exp_name):
    """Derive log file path inside the model folder."""
    return os.path.join(model_path, "seg_logs", f"{exp_name}.txt")
