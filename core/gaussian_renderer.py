#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import NamedTuple

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from core.gaussian_model import GaussianModel
from core.params import PipeParams
from core.scene.camera import Camera, ViewerCamera
from core.utils.sh_utils import eval_sh


class RenderPackage(NamedTuple):
    render: torch.Tensor | None
    features: torch.Tensor | None
    depth: torch.Tensor | None
    variance: torch.Tensor | None
    alpha: torch.Tensor | None
    viewspace_points: torch.Tensor
    visibility_filter: torch.BoolTensor
    radii: torch.Tensor


def render(
    viewpoint_camera: Camera | ViewerCamera,
    model: GaussianModel,
    pipe: PipeParams,
    bg_color: torch.Tensor,
    max_sh_degree: int,
    active_sh_degree: int | None = None,
    bg_features: torch.Tensor | None = None,
    scaling_modifier: float = 1.0,
    override_color: torch.Tensor | None = None,
    override_features: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
    render_color: bool = True,
    render_features: bool = True,
    render_depth: bool = False,
    render_var: bool = False,
    render_alpha: bool = False,
):
    """
    Render the scene.
    """

    assert bg_color.is_cuda, "background color tensor must be on GPU"
    assert bg_features is None or bg_features.is_cuda, (
        "background features tensor must be on GPU"
    )
    assert override_color is None or override_color.is_cuda, (
        "color tensor must be on GPU"
    )
    assert override_features is None or override_features.is_cuda, (
        "features tensor must be on GPU"
    )
    assert override_opacity is None or override_opacity.is_cuda, (
        "opacity tensor must be on GPU"
    )

    feats = model.get_features if override_features is None else override_features
    if bg_features is None:
        if feats is not None:
            bg_features = torch.zeros(
                (feats.shape[-1],), dtype=torch.float32, device=bg_color.device
            )
        else:
            bg_features = torch.empty((0,), dtype=torch.float32, device=bg_color.device)
    else:
        assert feats is None or bg_features.shape[-1] == feats.shape[-1], (
            "Background features must have the same dimension as the model features"
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            model.get_xyz,
            dtype=model.get_xyz.dtype,
            requires_grad=True,
            device=model._xyz.device,
        )
        + 0
    )

    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fovX * 0.5)
    tanfovy = math.tan(viewpoint_camera.fovY * 0.5)

    sh_degree = max_sh_degree if active_sh_degree is None else active_sh_degree

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        bg_features=bg_features,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        render_color=render_color,
        render_depth=render_depth,
        render_var=render_var,
        render_alpha=render_alpha,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = model.get_xyz
    means2D = screenspace_points

    if override_opacity is None:
        opacity = model.get_opacity
    else:
        opacity = override_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = model.get_covariance(scaling_modifier)
    else:
        scales = model.get_scaling
        rotations = model.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_shs_python:
            shs_view = model.get_spherical_harmonics.transpose(1, 2).view(
                -1, 3, (max_sh_degree + 1) ** 2
            )
            dir_pp = model.get_xyz - viewpoint_camera.camera_center.repeat(
                model.get_spherical_harmonics.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = model.get_spherical_harmonics
    else:
        colors_precomp = override_color

    model_features: torch.Tensor | None = None
    if render_features:
        if override_features is not None:
            model_features = override_features
        elif model.get_features is not None:
            model_features = model.get_features
    else:
        assert not render_var, (
            "Cannot render variance without features. Set render_features to True."
        )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (
        rendered_image,
        features,
        depth,
        var,
        alpha,
        radii,
    ) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        features=model_features,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return RenderPackage(
        render=rendered_image if render_color else None,
        features=features
        if render_features
        and (model.get_features is not None or override_features is not None)
        else None,
        depth=depth if render_depth else None,
        variance=var if render_var else None,
        alpha=alpha if render_alpha else None,
        viewspace_points=screenspace_points,
        visibility_filter=radii > 0,
        radii=radii,
    )
