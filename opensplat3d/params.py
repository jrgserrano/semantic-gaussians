from dataclasses import dataclass, field


@dataclass
class ModelParams:
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""

    images: str = "images"
    resolution: int = -1
    white_background: bool = False
    data_device: str = "cuda:1"
    eval: bool = False
    test_hold: int | float = 8
    init_points: int = 100_000
    init_type: str = "sample"  # random, points, sample
    init_ply: str | None = None

    num_frames: int = -1
    nth_frames: int = -1
    frames_dist: str = "uniform"

    mask_subdir: str | None = None
    mask_level: str = "default"
    mask_dim: int = 8


@dataclass
class PipeParams:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_shs_python: bool = False


@dataclass
class OptimizationParams:
    iterations: int = 30_000
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    sh_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    feature_lr: float = 2.5e-3

    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 2e-4
    random_background: bool = False

    only_features: bool = False
    num_points_limit: int = 0
    feature_init: str = "sh"
    static_xyz: bool = False
    random_background_features: bool = False

    photo_lambda: float = 1.0

    inst2d_lambda: float = 1.0
    inst2d_sample_size: int = -1
    inst2d_gamma: float = 1.0
    inst2d_weights: list[float] = field(default_factory=lambda: [1.0, 1.0])
    inst2d_normalize: bool = False
    inst2d_interval: int = 1
    inst2d_from_iter: int = 0

    var_lambda: float = 0.0
    lambda_erank: float = 0.0
    lambda_thin: float = 0.0
    
    # Depth parameters
    lambda_depth: float = 1
    depth_loss_type: str = "l1"

    # Advanced Pruning Suite (FeatureSLAM / LEGO-SLAM / OpenGS-SLAM)
    lambda_c: float = 0.5  # Weight for color gradient in importance scoring
    lambda_f: float = 0.5  # Weight for feature gradient in importance scoring
    tau_dist: float = 0.1  # Spatial distance threshold for redundancy
    tau_sim: float = 0.95  # CLIP similarity threshold for redundancy
    theta_scale: float = 0.25  # Scale threshold for boundary conflict pruning

    semantic_pruning_interval: int = 1000
    semantic_pruning_percentile: float = 0.1


@dataclass
class ClusterParams:
    enabled: bool = False
    output_dir: str | None = None
    position: float = 0.0
    color: float = 0.0
    min_size: int = 5
    min_samples: int | None = None
    eps: float = 0.0


@dataclass
class LanguageParams:
    enabled: bool = False
    model: str = "clip"
    topk: int = 5
    levels: int = 1
    ratio: float = 0.1
    dynamic_ratio: bool = False
    masked: bool = False
    rendering: bool = False
    pred_thresh: float = 0.2
    alpha_blend: float = 0.0


@dataclass
class ExportScanNetppParams:
    enabled: bool = False
    output_path: str | None = None
    knn_k: int = 1
    sem_topk: int = 3
    use_segments: bool = False


@dataclass
class DescriptionParams:
    enabled: bool = False
    vlm: str = "llava-hf/llava-1.5-7b-hf"
    topk: int = 3
    debug: bool = False
