from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from IPython.display import display as display_
from ipywidgets import HTML

from opensplat3d.gaussian_model import GaussianModel, sub_gaussians
from opensplat3d.gaussian_renderer import render
from opensplat3d.params import ModelParams, PipeParams
from opensplat3d.scene import Camera
from opensplat3d.utils.sh_utils import SH2RGB
from opensplat3d.utils.vis_utils import (
    enhance_image,
    feature_image_pca_3d,
    images2video,
)
from opensplat3d.utils.vis_utils import (
    pca as pca_,
)


def render_frames(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    override_color: torch.Tensor | None = None,
    override_features: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    bg_features: torch.Tensor | None = None,
    pca: str | None = None,
    normalize: bool = False,
    scaling_modifier: float = 1.0,
    downsample_factor: int = 1,
):
    pipe_params = PipeParams()
    if bg is None:
        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    if bg_features is None and gaussians.get_features is not None:
        dims = gaussians.get_features.size(-1)
        bg_features = torch.rand((dims), device=device)

    renders: list[npt.NDArray] = []
    visibility_filter: list[npt.NDArray] = []
    with torch.no_grad():
        if override_color is None and pca == "3d":
            assert gaussians.get_features is not None, (
                "features of Gaussians not available"
            )
            feats = gaussians.get_features[: model_params.mask_dim]
            override_color = (
                torch.from_numpy(
                    pca_(
                        feats.cpu().numpy(),
                        normalize=True,
                    )
                )
                .float()
                .to(device)
            )
        for cam in cameras:
            render_pkg = render(
                cam,
                gaussians,
                pipe_params,
                bg,
                model_params.sh_degree,
                bg_features=bg_features,
                override_color=override_color,
                override_features=override_features,
                override_opacity=override_opacity,
                scaling_modifier=scaling_modifier,
            )
            visibility_filter.append(render_pkg.visibility_filter.cpu().numpy())
            if pca is None or pca == "3d":
                assert render_pkg.render is not None, "Rendered image is None"
                image = (
                    render_pkg.render.permute(1, 2, 0)
                    .contiguous()
                    .clamp(0, 1)
                    .mul(255)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                renders.append(image[::downsample_factor, ::downsample_factor])
            else:
                assert render_pkg.features is not None, "features were not rendered"
                feats = render_pkg.features[: model_params.mask_dim]
                if normalize:
                    renders.append(
                        feature_image_pca_3d(
                            F.normalize(feats, dim=0)
                            .cpu()
                            .numpy()[:, ::downsample_factor, ::downsample_factor]
                        )
                    )
                else:
                    renders.append(
                        feature_image_pca_3d(feats.cpu().numpy())[
                            :, ::downsample_factor, ::downsample_factor
                        ]
                    )
    return np.stack(renders), np.stack(visibility_filter)


def render_features(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    override_color: torch.Tensor | None = None,
    override_features: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
    downsample_factor: int = 1,
):
    pipe_params = PipeParams()
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    assert gaussians.get_features is not None, "features of Gaussians not available"
    dims = gaussians.get_features.size(-1)
    bg_features = torch.rand((dims), device=device)
    renders: list[torch.Tensor] = []
    with torch.no_grad():
        for cam in cameras:
            render_pkg = render(
                cam,
                gaussians,
                pipe_params,
                bg,
                model_params.sh_degree,
                bg_features=bg_features,
                override_color=override_color,
                override_features=override_features,
                override_opacity=override_opacity,
            )
            assert render_pkg.features is not None
            renders.append(
                render_pkg.features[:, ::downsample_factor, ::downsample_factor].cpu()
            )
    return torch.stack(renders)


def render_video(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    override_color: torch.Tensor | None = None,
    override_features: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    bg_features: torch.Tensor | None = None,
    pca: str | None = None,
    normalize: bool = False,
    enhance_pca: bool = False,
    display: bool = True,
    scaling_modifier: float = 1.0,
    downsample_factor: int = 2,
    interval: int = 100,
):
    renders, _ = render_frames(
        cameras,
        gaussians,
        model_params,
        device,
        override_color,
        override_features,
        override_opacity,
        bg,
        bg_features,
        pca,
        normalize,
        scaling_modifier=scaling_modifier,
    )
    if pca == "3d" and enhance_pca:
        renders = np.stack([enhance_image(x, 3.0, 2.0) for x in renders])
    video = images2video(
        renders[:, ::downsample_factor, ::downsample_factor],
        interval=interval,
        dpi=72.0,
    )
    if display:
        display_(HTML(video.to_html5_video()))
    return video


def render_cluster_frames(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    mask: torch.BoolTensor | torch.LongTensor,
    override_color: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    reset_opacity: bool = False,
    downsample_factor: int = 1,
):
    if override_color is not None:
        override_color = override_color[mask]
    cluster_gaussians = sub_gaussians(gaussians, mask)
    if reset_opacity:
        cluster_gaussians._opacity = cluster_gaussians.inverse_opacity_activation(
            torch.ones_like(cluster_gaussians._opacity)
        )
    return render_frames(
        cameras,
        cluster_gaussians,
        model_params,
        device,
        override_color,
        bg=bg,
        downsample_factor=downsample_factor,
    )


def render_cluster(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    mask: torch.BoolTensor | torch.LongTensor,
    override_color: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    reset_opacity: bool = False,
    display: bool = True,
):
    renders, _ = render_cluster_frames(
        cameras,
        gaussians,
        model_params,
        device,
        mask,
        override_color,
        bg,
        reset_opacity,
    )
    video = images2video(
        renders[:, ::2, ::2],
        interval=100,
        dpi=72.0,
    )
    if display:
        display_(HTML(video.to_html5_video()))
    return video


def render_mask_frames(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    mask: torch.BoolTensor | torch.LongTensor,
    cluster_color: torch.Tensor,
    bg: torch.Tensor | None = None,
    downsample_factor: int = 1,
):
    # use only basis color of Gaussians
    override_color = SH2RGB(
        gaussians.get_spherical_harmonics.detach().clone()[..., 0, :]
    )
    override_color[mask] = cluster_color[mask]
    return render_frames(
        cameras,
        gaussians,
        model_params,
        device,
        override_color,
        bg,
        downsample_factor=downsample_factor,
    )


def render_mask(
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    device: torch.device,
    mask: torch.BoolTensor | torch.LongTensor,
    cluster_color: torch.Tensor,
    bg: torch.Tensor | None = None,
    display: bool = True,
):
    renders, _ = render_mask_frames(
        cameras,
        gaussians,
        model_params,
        device,
        mask,
        cluster_color,
        bg,
    )
    video = images2video(
        renders[:, ::2, ::2],
        interval=100,
        dpi=72.0,
    )
    if display:
        display_(HTML(video.to_html5_video()))
    return video


def load_labels(model_path: Path) -> tuple[npt.NDArray | None, npt.NDArray | None]:
    if (model_path / "clustering" / "labels.npy").exists():
        labels = np.load(model_path / "clustering" / "labels.npy")
        labels_, counts = np.unique(labels, return_counts=True)
        sort_index = np.argsort(counts)[::-1]
        labels_ = labels_[sort_index]
    else:
        labels = None
        labels_ = None

    return labels, labels_
