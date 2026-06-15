import datetime
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from core.config import Config, config_from_yaml
from core.gaussian_model import GaussianModel
from core.gaussian_renderer import render
from core.params import ModelParams, PipeParams
from core.scene import Camera


# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # type: ignore
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


@torch.no_grad()
def render_frames(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    override_color: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
    render_features: bool = True,
):
    pipe_params = PipeParams()
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    rendered_images: list[torch.Tensor] = []
    rendered_features: list[torch.Tensor] = []

    for cam in cameras:
        render_pkg = render(
            cam,
            gaussians,
            pipe_params,
            bg,
            model_params.sh_degree,
            override_color=override_color,
            override_opacity=override_opacity,
            render_features=render_features,
        )
        assert render_pkg.render is not None, "Rendered image is None"
        image = render_pkg.render.clamp(0, 1).cpu()
        rendered_images.append(image)
        if render_pkg.features is not None:
            rendered_features.append(render_pkg.features.cpu())

    assert len(rendered_features) == 0 or len(rendered_images) == len(
        rendered_features
    ), (
        f"Number of rendered images ({len(rendered_images)}) "
        f"does not match number of rendered features ({len(rendered_features)})"
    )
    return torch.stack(rendered_images), torch.stack(rendered_features) if len(
        rendered_features
    ) > 0 else None


@torch.no_grad()
def render_alpha(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    override_opacity: torch.Tensor | None = None,
):
    pipe_params = PipeParams()
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    rendered_alpha: list[torch.Tensor] = []

    for cam in cameras:
        render_pkg = render(
            cam,
            gaussians,
            pipe_params,
            bg,
            model_params.sh_degree,
            override_opacity=override_opacity,
            render_color=False,
            render_features=False,
            render_alpha=True,
        )
        assert render_pkg.alpha is not None, "Rendered alpha is None"
        rendered_alpha.append(1 - render_pkg.alpha.cpu())

    return torch.stack(rendered_alpha)


def get_scene_model_configs(model_paths: list[Path]):
    # For lerf-mask evaluation
    scene_model_configs: dict[str, list[tuple[Path, Config]]] = {}
    for model_path in model_paths:
        config = config_from_yaml(model_path / "config.yaml")
        source_path = Path(config.model.source_path)
        scene_name = source_path.stem
        if scene_name not in scene_model_configs:
            scene_model_configs[scene_name] = []
        scene_model_configs[scene_name].append((model_path, config))
    return scene_model_configs


def generate_eval_output_name():
    uid = str(uuid.uuid4())[:8]
    dtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{dtime}-{uid}"
