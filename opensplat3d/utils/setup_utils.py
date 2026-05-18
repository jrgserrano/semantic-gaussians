from dataclasses import dataclass
from pathlib import Path

import torch

from opensplat3d.config import Config, config_from_yaml
from opensplat3d.data import load_scene_info
from opensplat3d.gaussian_model import GaussianModel, create_from_ply
from opensplat3d.params import ModelParams
from opensplat3d.scene.scene import Scene
from opensplat3d.utils.scene_utils import SceneInfo, search_for_max_iteration


@dataclass(frozen=True)
class ModelSetupParams:
    model_path: Path
    config: Config
    model_params: ModelParams
    device: torch.device
    iteration: int
    gaussians: GaussianModel


@dataclass(frozen=True)
class SetupParams(ModelSetupParams):
    scene_info: SceneInfo
    scene: Scene


def get_latest_model(model_path: Path):
    dirs = [p for p in model_path.iterdir() if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: p.stem.split("-")[0])
    return dirs[-1]


def setup_model_from_config(
    config: Config, model_path: Path | None = None, iteration: int | None = None
):
    model_params = config.model
    model_path = Path(model_params.model_path) if model_path is None else model_path
    
    device = torch.device(model_params.data_device)
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda:0')
        elif device.index >= torch.cuda.device_count():
            print(f"Warning: Device {device} not found. Falling back to cuda:0")
            device = torch.device('cuda:0')
    
    if iteration is None:
        iteration = search_for_max_iteration(model_path / "point_cloud")
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    gaussians = create_from_ply(
        ply_path, model_params.sh_degree, device, feature_dim=model_params.mask_dim
    )
    feature_dim = (
        gaussians.get_features.size(1) if gaussians.get_features is not None else None
    )
    print(
        f"Iteration {iteration} | num_gaussians: {gaussians.num_points} | feature_dim: {feature_dim}"
    )
    return ModelSetupParams(
        model_path, config, model_params, device, iteration, gaussians
    )


def setup_model(model_path: Path, iteration: int | None = None):
    if not (model_path / "config.yaml").exists():
        model_path = get_latest_model(model_path)
    config = config_from_yaml(model_path / "config.yaml")
    return setup_model_from_config(config, model_path, iteration)


def setup_from_config(
    config: Config, model_path: Path | None = None, iteration: int | None = None
):
    model_setup_params = setup_model_from_config(config, model_path, iteration)
    scene_info = load_scene_info(model_setup_params.model_params)
    scene = Scene(
        scene_info,
        model_setup_params.model_params.resolution,
        model_setup_params.device,
        shuffle=False,
    )
    return SetupParams(
        model_setup_params.model_path,
        model_setup_params.config,
        model_setup_params.model_params,
        model_setup_params.device,
        model_setup_params.iteration,
        model_setup_params.gaussians,
        scene_info,
        scene,
    )


def setup(
    model_path: Path,
    mask_subdir: str | None = None,
    load_masks: bool = True,
    num_frames: int | None = None,
    iteration: int | None = None,
):
    if not (model_path / "config.yaml").exists():
        model_path = get_latest_model(model_path)
    config = config_from_yaml(model_path / "config.yaml")
    model_params = config.model
    if mask_subdir is not None:
        model_params.mask_subdir = mask_subdir
    if not load_masks:
        model_params.mask_subdir = None
    if num_frames is not None:
        assert num_frames >= -1, "Number of frames must be greater or equal to -1"
        model_params.num_frames = num_frames
    return setup_from_config(config, model_path, iteration)
