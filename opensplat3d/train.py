import os
import sys

# Force use of only GPU 0 (RTX 3060 Ti) to avoid multi-GPU context/memory issues with TITAN X
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import typing
from pathlib import Path

import torch
import wandb
from diff_gaussian_rasterization import MAX_FEATURE_DIM
from tqdm import tqdm

from opensplat3d.cluster.hdbscan import hdbscan_clustering
from opensplat3d.data import load_scene_info
from opensplat3d.eval.scannetpp.export_predictions import export_scenes_predictions
from opensplat3d.gaussian_model import create_from_pcd, create_from_ply, save_gaussians
from opensplat3d.gaussian_optimizer import GaussianOptimizer
from opensplat3d.gaussian_renderer import render
from opensplat3d.language import LanguageModel
from opensplat3d.language.embed import embed
from opensplat3d.semantic.descriptions import compute_descriptions, CropParams
from opensplat3d.utils.setup_utils import setup
from opensplat3d.losses import (
    get_erank_loss,
    get_thinness_loss,
    instance_2d_loss,
    l1_loss,
    ssim,
)
from opensplat3d.params import ModelParams, OptimizationParams, PipeParams
from opensplat3d.scene import Scene
from opensplat3d.utils.general_utils import seed_everything
from opensplat3d.utils.scene_utils import save_scene_info
from opensplat3d.utils.train_utils import (
    FisherCameraSampler,
    save_quality_heatmaps,
    setup_training,
    train_cameras,
    training_report,
)

import numpy as np
from opensplat3d.utils.general_utils import build_rotation

def get_gt_normals(viewpoint_cam, device):
    """Calcula el mapa de normales GT (World Space) desde la profundidad."""
    depth = viewpoint_cam.original_depth.to(device)
    h, w = depth.shape
    fx = w / (2 * np.tan(viewpoint_cam.fovX / 2))
    fy = h / (2 * np.tan(viewpoint_cam.fovY / 2))
    cx, cy = w / 2, h / 2

    i, j = torch.meshgrid(torch.linspace(0, w-1, w, device=device), 
                          torch.linspace(0, h-1, h, device=device), indexing='xy')
    
    points = torch.stack([(i - cx) * depth / fx, (j - cy) * depth / fy, depth], dim=-1)
    
    dy = torch.zeros_like(points)
    dx = torch.zeros_like(points)
    dy[1:-1, :, :] = points[2:, :, :] - points[:-2, :, :]
    dx[:, 1:-1, :] = points[:, 2:, :] - points[:, :-2, :]

    normals = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    
    # Pasar a World Space
    R_c2w = viewpoint_cam.R.to(device)
    normals_world = (R_c2w @ normals.view(-1, 3).T).T.view(h, w, 3)
    return normals_world.permute(2, 0, 1) # (3, H, W)

def render_normals(viewpoint_cam, gaussians, pipe, config, device):
    """Renderiza el mapa de normales actual (World Space)."""
    R = build_rotation(gaussians._rotation)
    scales = gaussians.get_scaling
    min_scale_idx = torch.argmin(scales, dim=1)
    
    normals = torch.gather(R, 2, min_scale_idx.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
    normal_colors = normals * 0.5 + 0.5 # Mapeo a [0, 1] para el renderizador
    
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    res = render(viewpoint_cam, gaussians, pipe, bg, config.model.sh_degree,
                 config.model.sh_degree, override_color=normal_colors)
    
    # Devolver mapeado de vuelta a [-1, 1] para el cálculo del Loss
    return res.render * 2.0 - 1.0

def normal_consistency_loss(n_render, n_gt):
    """
    Calcula la pérdida de normales basada en similitud de coseno.
    n_render: (3, H, W) en rango [-1, 1]
    n_gt: (3, H, W) en rango [-1, 1]
    """
    # Producto escalar entre vectores normales (Cosine Similarity)
    # 1.0 significa que apuntan al mismo sitio, -1.0 al contrario.
    cos_sim = (n_render * n_gt).sum(dim=0)
    
    # Queremos que la similitud sea 1.0 (o -1.0 ya que la normal de un 'disco' es ambigua)
    # Por eso usamos el valor absoluto de la similitud
    loss = 1.0 - torch.abs(cos_sim)
    
    return loss.mean()


def training(
    config,
    model_params: ModelParams,
    opt_params: OptimizationParams,
    pipe_params: PipeParams,
    save_iterations: list[int],
    checkpoint_iterations: list[int],
    test_iterations: list[int],
    checkpoint_path: Path | None = None,
):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    # validate config, e.g. dimensions
    mask_dim = model_params.mask_dim
    assert mask_dim <= MAX_FEATURE_DIM, (
        f"Feature dimension exceeds limit of {MAX_FEATURE_DIM} for mask={mask_dim}"
    )
    if model_params.mask_subdir is not None:
        print(f"Using {mask_dim}/{MAX_FEATURE_DIM} dimensions, mask={mask_dim}")
        print(f"Using mask level: {model_params.mask_level}")

    model_path: Path = Path(model_params.model_path)
    
    # Initialize history tracking
    history = {
        "loss": [],
        "psnr_test": [],
        "psnr_train": [],
        "ssim_test": [],
        "ssim_train": [],
        "miou_test": 0.0,
        "mbiou_test": 0.0,
        "miou_train": 0.0,
        "mbiou_train": 0.0,
        "feature_instability": [],
    }
    stats_path = model_path / "training_stats.json"

    # Ensure PSNR is calculated every 1000 iterations
    for i in range(1000, opt_params.iterations + 1, 1000):
        if i not in test_iterations:
            test_iterations.append(i)
    test_iterations.sort()
    scene_info = load_scene_info(model_params)
    save_scene_info(scene_info, model_path)

    if model_params.data_device is None or model_params.data_device == "cuda:1":
        model_params.data_device = "cuda:0"
    
    device = torch.device(model_params.data_device)

    scene = Scene(scene_info, model_params.resolution, device)
    assert scene_info.point_cloud is not None, "Point cloud is required"

    if opt_params.static_xyz:
        print(
            "Freezing XYZ coordinates: parameters will receive no gradients and are additionally removed from the optimizer"
        )

    only_features = False
    ply_init_path: Path | None = (
        Path(model_params.init_ply) if model_params.init_ply is not None else None
    )
    if ply_init_path is not None:
        print(f"Initializing from {ply_init_path}")
        only_features = opt_params.only_features
        if only_features:
            print("Freezing all parameters except for the features.")
        gaussians = create_from_ply(
            ply_init_path,
            model_params.sh_degree,
            device,
            only_features,
            mask_dim,
        )
    else:
        gaussians = create_from_pcd(
            scene_info.point_cloud,
            model_params.sh_degree,
            device,
            model_params.mask_subdir is not None,
            mask_dim,
            opt_params.feature_init,
            opt_params.static_xyz,
        )

    first_iter = 0
    optimizer = GaussianOptimizer(
        gaussians,
        opt_params,
        model_params.sh_degree,
        scene.cameras_extent,
        opt_params.static_xyz,
        device,
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        first_iter = checkpoint["iteration"]
        gaussians.load_state_dict(checkpoint["gaussians"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resuming training from iteration {first_iter}")

    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    loss_inst2d_start = torch.cuda.Event(enable_timing=True)
    loss_inst2d_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        range(first_iter, opt_params.iterations), desc="Training progress"
    )
    first_iter += 1
    
    # New: Active Learning Sampler
    all_cameras = scene.get_train_cameras()
    sampler = FisherCameraSampler(all_cameras)

    pruning_history = []
    pruning_log_path = model_path / "pruning_log.json"

    for iteration in range(first_iter, opt_params.iterations + 1):
        iter_start.record()  # type: ignore

        optimizer.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            optimizer.oneup_sh_degree()

        viewpoint_cam = sampler.sample()

        render_color = opt_params.photo_lambda > 0
        render_features = gaussians.get_features is not None and mask_dim > 0
        render_var = render_features and opt_params.var_lambda > 0

        # Enable depth rendering if we have depth maps and lambda > 0
        has_depth = hasattr(viewpoint_cam, "original_depth") and viewpoint_cam.original_depth is not None
        render_depth = has_depth and opt_params.lambda_depth > 0

        bg = (
            torch.rand((3), device=device)
            if opt_params.random_background
            else background
        )
        bg_features = (
            torch.rand((mask_dim), device=device)
            if opt_params.random_background_features
            else None
        )

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe_params,
            bg,
            model_params.sh_degree,
            optimizer.active_sh_degree,
            bg_features=bg_features,
            render_color=render_color,
            render_features=render_features,
            render_depth=render_depth,
            render_var=render_var,
        )

        rendered_features = render_pkg.features
        rendered_variance = render_pkg.variance
        rendered_depth = render_pkg.depth

        # Importance Tracking (FeatureSLAM)
        # We need separate gradients for color and features
        opacity_grad_rgb = None
        opacity_grad_feat = None

        # Loss
        loss = torch.tensor(0.0, device=device)

        Ll1: torch.Tensor | None = None
        photometric_loss: torch.Tensor | None = None
        if not only_features:
            if render_pkg.render is not None:
                gt_image = viewpoint_cam.original_image.to(device)
                Ll1 = l1_loss(render_pkg.render, gt_image)
                photometric_loss = (
                    1.0 - opt_params.lambda_dssim
                ) * Ll1 + opt_params.lambda_dssim * (
                    1.0 - ssim(render_pkg.render, gt_image)
                )
                loss += opt_params.photo_lambda * photometric_loss
        else:
            assert rendered_features is not None, (
                "Features are required if optimizing features only"
            )

        # Depth Loss
        depth_loss: torch.Tensor | None = None
        if render_depth and rendered_depth is not None and has_depth:
            gt_depth = viewpoint_cam.original_depth.to(device)
            depth_loss = l1_loss(rendered_depth, gt_depth)
            loss += opt_params.lambda_depth * depth_loss
        
        # Normal Consistency Loss
        """
        lambda_normal = 0.05

        normals = render_normals(viewpoint_cam, gaussians, pipe_params, config, device)
        normals_gt = get_gt_normals(viewpoint_cam, device)
        loss_norm = normal_consistency_loss(normals, normals_gt)
        loss += lambda_normal * loss_norm
        """

        # Variance loss
        var_loss: torch.Tensor | None = None
        if render_var:
            assert rendered_variance is not None, (
                "Variance is required for variance loss"
            )
            var_loss = rendered_variance[:mask_dim].pow(2).mean()
            loss += opt_params.var_lambda * var_loss

        # Instance 2D loss
        inst2d_loss: dict[str, torch.Tensor] | None = None
        if (
            viewpoint_cam.masks is not None
            and rendered_features is not None
            and (
                opt_params.inst2d_interval > 0
                and iteration > opt_params.inst2d_from_iter
                and iteration % opt_params.inst2d_interval == 0
            )
        ):
            gt_masks = viewpoint_cam.masks.to(device)
            loss_inst2d_start.record()  # type: ignore
            inst2d_loss = instance_2d_loss(
                rendered_features[:mask_dim],
                gt_masks,
                model_params.mask_dim,
                opt_params.inst2d_sample_size,
                opt_params.inst2d_gamma,
                opt_params.inst2d_weights,
                opt_params.inst2d_normalize,
            )
            loss_inst2d_end.record()  # type: ignore
            total_inst2d_loss = opt_params.inst2d_lambda * inst2d_loss["total"]
            loss += total_inst2d_loss
            
            # Feature Component of Importance
            try:
                opacity_grad_feat = torch.autograd.grad(total_inst2d_loss, gaussians._opacity, retain_graph=True, allow_unused=True)[0]
            except Exception:
                pass
        
        # Geometric losses (FeatureSLAM)
        """
        warmup_weight = min(1.0, iteration / 2000)

        erank_loss: torch.Tensor | None = None
        if opt_params.lambda_erank > 0 and warmup_weight > 0 and iteration % 10 == 0:
            erank_loss = get_erank_loss(gaussians.get_scaling)
            loss += warmup_weight * opt_params.lambda_erank * erank_loss

        thin_loss: torch.Tensor | None = None
        if opt_params.lambda_thin > 0 and warmup_weight > 0 and iteration % 10 == 0:
            thin_loss = get_thinness_loss(gaussians.get_scaling)
            loss += warmup_weight * opt_params.lambda_thin * thin_loss
        """
        # RGB Component of Importance
        if Ll1 is not None and not only_features:
            try:
                opacity_grad_rgb = torch.autograd.grad(opt_params.photo_lambda * photometric_loss, gaussians._opacity, retain_graph=True, allow_unused=True)[0]
            except Exception:
                pass

        # Component-wise gradients for Smart Refinement
        xyz_grad_rgb = None
        xyz_grad_sem = None
        
        if not only_features and photometric_loss is not None:
            try:
                xyz_grad_rgb = torch.autograd.grad(opt_params.photo_lambda * photometric_loss, render_pkg.viewspace_points, retain_graph=True, allow_unused=True)[0]
            except Exception:
                pass
        
        if inst2d_loss is not None:
            try:
                xyz_grad_sem = torch.autograd.grad(opt_params.inst2d_lambda * inst2d_loss["total"], render_pkg.viewspace_points, retain_graph=True, allow_unused=True)[0]
            except Exception:
                pass

        loss.backward()

        # Record loss history
        history["loss"].append({"iter": iteration, "value": loss.item()})

        optimizer.add_stats(render_pkg.viewspace_points, render_pkg.visibility_filter, render_pkg.radii, xyz_grad_rgb, xyz_grad_sem)
        optimizer.add_importance_stats(opacity_grad_rgb, opacity_grad_feat, opt_params)

        # Update Smart Sampler Score using EIG approximation
        # Surprise = (current_grad^2) / (accumulated_fim + eps)
        with torch.no_grad():
            visible_mask = render_pkg.visibility_filter
            curr_grad2 = (render_pkg.viewspace_points.grad[visible_mask, :2] ** 2).sum(dim=-1, keepdim=True)
            accum_fim = optimizer.fim_accum[visible_mask]
            surprise = (curr_grad2 / (accum_fim + 0.01)).sum().item()
            sampler.update_score(viewpoint_cam.uid, surprise)

        iter_end.record()  # type: ignore

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            timings = {}
            if iteration % 10 == 0:
                postfix = {
                    "loss": f"{ema_loss_for_log:.7f}",
                }
                if inst2d_loss is not None:
                    postfix["inst2d"] = f"{inst2d_loss['total']:.4e}"
                    if photometric_loss is not None:
                        postfix["photo"] = f"{photometric_loss:.7f}"
                if var_loss is not None:
                    postfix["var"] = f"{var_loss:.4e}"
                if depth_loss is not None:
                    postfix["depth"] = f"{depth_loss:.4e}"
                #if erank_loss is not None:
                #    postfix["erank"] = f"{erank_loss:.4e}"
                #if thin_loss is not None:
                #    postfix["thin"] = f"{thin_loss:.4e}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)

                timings = {
                    "iter": iter_start.elapsed_time(iter_end),
                }

                losses = {
                    "total_loss": {"value": loss, "log": True},
                    "photometric_loss": {
                        "value": photometric_loss,
                        "log": True,
                    },
                    "variance_loss": {"value": var_loss, "log": True},
                    #"erank_loss": {"value": erank_loss, "log": True},
                    #"thinness_loss": {"value": thin_loss, "log": True},
                }
                if inst2d_loss is not None:
                    for k, v in inst2d_loss.items():
                        losses[f"instance2d_loss/{k}"] = {
                            "value": v,
                            "log": True,
                            "timing": loss_inst2d_start.elapsed_time(loss_inst2d_end)
                            if k == "total"
                            else None,
                        }

                # Log and save
                report_results = training_report(
                    iteration,
                    timings.get("iter", None),
                    Ll1,
                    losses,
                    test_iterations,
                    gaussians,
                    scene,
                    pipe_params,
                    background,
                    model_params.sh_degree,
                    optimizer.active_sh_degree,
                    optimizer,
                )

                # Diagnostic Heatmaps
                if iteration % 1000 == 0:
                    try:
                        if len(scene.get_test_cameras()) > 0:
                            viewpoint_report = scene.get_test_cameras()[0]
                        else:
                            viewpoint_report = scene.get_train_cameras()[0]
                        render_report = render(viewpoint_report, gaussians, pipe_params, background, model_params.sh_degree, optimizer.active_sh_degree)
                        save_quality_heatmaps(
                            iteration, 
                            model_path, 
                            viewpoint_report.name, 
                            render_report.render, 
                            render_report.features, 
                            viewpoint_report.original_image.cuda(), 
                            viewpoint_report.masks.cuda() if viewpoint_report.masks is not None else None,
                            model_params.mask_dim
                        )
                    except Exception as e:
                        print(f"[ WARNING ] Diagnostic generation failed: {e}")

                for key in ["psnr_test", "psnr_train", "ssim_test", "ssim_train", "feature_instability"]:
                    if key in report_results:
                        history[key].append({"iter": iteration, "value": report_results[key]})

                # Periodically save history to file
                if iteration % 1000 == 0 or iteration == opt_params.iterations:
                    import json
                    with open(stats_path, "w") as f:
                        json.dump(history, f, indent=4)

            if iteration == opt_params.iterations:
                progress_bar.close()

            if iteration in save_iterations or iteration == opt_params.iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians ({gaussians.num_points})")
                save_gaussians(gaussians, model_path, iteration)

            # --- Progressive Cooldown Logic ---
            cooldown_start = 10000
            cooldown_end = 18000
            
            # Linear progress factor (0.0 to 1.0)
            cooldown_progress = max(0.0, min(1.0, (iteration - cooldown_start) / (cooldown_end - cooldown_start))) if iteration > cooldown_start else 0.0
            
            # Adaptive Intervals: Frequencies decrease as we cool down
            # 100 -> 500
            curr_densify_interval = int(opt_params.densification_interval * (1 + 4 * cooldown_progress))
            # 1000 -> 3000
            curr_prune_interval = int(opt_params.semantic_pruning_interval * (1 + 2 * cooldown_progress))
            # 0.0002 -> 0.0006 (less sensitive)
            curr_grad_threshold = opt_params.densify_grad_threshold * (1 + 2 * cooldown_progress)
            # 0.05 -> 0.01 (less aggressive)
            curr_percentile = opt_params.semantic_pruning_percentile * (1.0 - 0.8 * cooldown_progress)
            
            # Densification
            if iteration < cooldown_end:
                optimizer.add_stats(
                    render_pkg.viewspace_points,
                    render_pkg.visibility_filter,
                    render_pkg.radii,
                )

                if (
                    iteration > opt_params.densify_from_iter
                    and iteration % curr_densify_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt_params.opacity_reset_interval else None
                    )
                    if (
                        opt_params.num_points_limit > 0
                        and opt_params.num_points_limit < gaussians.num_points
                    ):
                        with torch.no_grad():
                            max_opacity = gaussians.get_opacity.max()
                            to_prune = max(
                                gaussians.num_points - opt_params.num_points_limit,
                                0,
                            )
                            thresholds = torch.linspace(
                                0.005, 0.9 * max_opacity, 100, device=device
                            )
                            threshold_index = (
                                (
                                    (gaussians.get_opacity < thresholds).sum(dim=0)
                                    - to_prune
                                )
                                .abs()
                                .argmin()
                            )
                            opacity_threshold = typing.cast(
                                float, thresholds[threshold_index].item()
                            )
                    else:
                        opacity_threshold = 0.005

                    optimizer.densify_and_prune(
                        curr_grad_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )

                    # New: Semantic instability densification
                    if opt_params.lambda_instability_densify > 0:
                        num_split = optimizer.densify_and_split_by_instability(
                            opt_params.lambda_instability_densify,
                            scene.cameras_extent
                        )
                        if num_split > 0:
                            print(f"\n[ITER {iteration}] Semantic Split: {num_split} points (Threshold: {curr_grad_threshold:.5f})")

                    # Smart Refinement (New)
                    optimizer.smart_refine(iteration, opt_params, scene.cameras_extent)

                    if opt_params.semantic_pruning_interval > 0 and iteration % curr_prune_interval == 0:
                        # LEGO-SLAM Redundancy check (every 2 intervals)
                        if iteration % (opt_params.semantic_pruning_interval * 2) == 0:
                            stats_lego = optimizer.redundancy_pruning(opt_params.tau_dist, opt_params.tau_sim)
                            stats_lego["reason"] = "Redundancy"
                            stats_lego["iteration"] = iteration
                            pruning_history.append(stats_lego)

                        # Semantic Importance Pruning (Percentile based)
                        stats_sem = optimizer.semantic_pruning(curr_percentile)
                        stats_sem["reason"] = "Semantic"
                        stats_sem["iteration"] = iteration
                        pruning_history.append(stats_sem)

                        # OpenGS-SLAM Boundary check
                        stats_ogs = optimizer.scale_guided_pruning(opt_params.theta_scale, opt_params.theta_ratio)
                        stats_ogs["reason"] = "ScaleBoundary"
                        stats_ogs["iteration"] = iteration
                        pruning_history.append(stats_ogs)
                        
                        # New: Statistical Outlier Removal
                        if opt_params.sor_interval > 0 and iteration % (curr_prune_interval * 2) == 0:
                            stats_sor = optimizer.statistical_outlier_removal_pruning(opt_params.sor_k, opt_params.sor_std_ratio)
                            stats_sor["reason"] = "SOR_Floaters"
                            stats_sor["iteration"] = iteration
                            pruning_history.append(stats_sor)

                        # Save Log
                        import json
                        with open(pruning_log_path, "w") as f:
                            json.dump(pruning_history, f, indent=4)

                        # Reset tracking after pruning
                        optimizer.importance_accum.fill_(0)
                        optimizer.denom.fill_(0)

                if iteration % opt_params.opacity_reset_interval == 0 or (
                    model_params.white_background
                    and iteration == opt_params.densify_from_iter
                ):
                    optimizer.reset_opacity()

            # Optimizer step
            if iteration < opt_params.iterations:
                feat_old = None
                if gaussians._features is not None:
                    feat_old = gaussians._features.detach().clone()

                optimizer.optimizer.step()

                if feat_old is not None:
                    optimizer.record_feature_delta(feat_old, gaussians._features)

                optimizer.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving checkpoint")
                ckpt = {
                    "iteration": iteration,
                    "gaussians": gaussians.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                ckpt_dir = model_path / "ckpts"
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                torch.save(ckpt, ckpt_dir / f"{iteration}.pth")

    print(f"\nModel can be found at: {model_path}")
    return gaussians, scene, device, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training of an OpenSplat3D model")
    parser.add_argument("--config", type=str, help="Path to a config yaml file")
    parser.add_argument(
        "--detect_anomaly", action="store_true", help="Detect anomaly in autograd"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--wandb", type=str, help="Use wandb for logging")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to resume from")
    parser.add_argument(
        "--save_iterations",
        type=int,
        nargs="+",
        default=[7000, 30_000],
        help="Save iterations",
    )
    parser.add_argument(
        "--test_iterations",
        type=int,
        nargs="+",
        default=[7000, 30_000],
        help="Test iterations",
    )
    parser.add_argument(
        "--checkpoint_iterations",
        type=int,
        nargs="+",
        default=[],
        help="Checkpoint iterations",
    )
    parser.add_argument("overrides", nargs="*", help="Overrides for the config")
    args = parser.parse_args()

    config = setup_training(args)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # type: ignore

    seed_everything(args.seed)

    save_iterations: list[int] = args.save_iterations
    checkpoint_iterations: list[int] = args.checkpoint_iterations
    test_iterations: list[int] = args.test_iterations
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else None

    gaussians, scene, device, history = training(
        config,
        config.model,
        config.opt,
        config.pipe,
        save_iterations,
        checkpoint_iterations,
        test_iterations,
        checkpoint_path,
    )

    if config.cluster.enabled:
        model_path: Path = Path(config.model.model_path)
        output_dir = (
            Path(config.cluster.output_dir)
            if config.cluster.output_dir is not None
            else model_path / "clustering"
        )
        print("\n\n")
        print("Clustering")
        hdbscan_clustering(
            model_path,
            output_dir,
            config.cluster.position,
            config.cluster.color,
            config.cluster.min_size,
            config.cluster.min_samples,
            config.cluster.eps,
        )
        """
        # Record mIoU and mBIoU metrics after clustering
        print("Evaluating final semantic metrics (mIoU/mBIoU)...")
        from opensplat3d.eval.instance_eval import evaluate_replica_instance_metrics
        labels_path = output_dir / "labels.npy"
        if labels_path.exists():
            labels = torch.from_numpy(np.load(labels_path))
            # Test metrics
            miou_test, mbiou_test = evaluate_replica_instance_metrics(gaussians, scene, scene.get_test_cameras(), labels, device)
            # Train metrics (subset for speed)
            train_cameras_subset = [scene.get_train_cameras()[i] for i in range(0, len(scene.get_train_cameras()), 10)]
            miou_train, mbiou_train = evaluate_replica_instance_metrics(gaussians, scene, train_cameras_subset, labels, device)
            
            history["miou_test"] = float(miou_test)
            history["mbiou_test"] = float(mbiou_test)
            history["miou_train"] = float(miou_train)
            history["mbiou_train"] = float(mbiou_train)
            
            print(f"Final Test mIoU: {miou_test:.4f}, mBIoU: {mbiou_test:.4f}")
            print(f"Final Train mIoU: {miou_train:.4f}, mBIoU: {mbiou_train:.4f}")

        # Final save of stats
        with open(stats_path, "w") as f:
            import json
            json.dump(history, f, indent=4)
        """

        # language is based on clustering
        if config.lang.enabled:
            print("\n\n")
            print("Language Embeddings")
            embed(
                model_path,
                config.lang.model,
                config.lang.topk,
                config.lang.levels,
                config.lang.masked,
                config.lang.ratio,
                config.lang.dynamic_ratio,
                config.lang.alpha_blend,
                config.lang.rendering,
                config.lang.pred_thresh,
            )
            print("\n\n")

            if config.desc.enabled:
                print("Generating Object Descriptions")
                lang_model = LanguageModel(config.lang.model)
                crop_params = CropParams(
                    lang_model.img_size,
                    config.lang.levels,
                    config.lang.masked,
                    config.lang.ratio,
                    config.lang.dynamic_ratio,
                    config.lang.alpha_blend,
                )
                setup_params = setup(model_path)
                compute_descriptions(
                    setup_params,
                    lang_model,
                    crop_params,
                    config.lang.rendering,
                    config.lang.pred_thresh,
                    config.desc.topk,
                    config.desc.vlm,
                    config.desc.debug,
                )
                print("\n\n")

            if config.export_scannetpp.enabled:
                print("\n\n")
                print("Exporting scene to ScanNet++")
                export_subdir = "without-postprocessing"
                if config.export_scannetpp.use_segments:
                    export_subdir = "segments"
                export_subdir = f"eval_predictions/{export_subdir}"

                is_eval_run = model_path.parts[-3] == "scenes"

                if config.export_scannetpp.output_path is not None:
                    output_path = Path(config.export_scannetpp.output_path)
                elif is_eval_run:
                    # output/eval/scannetpp/{exp_id}/scenes/{scene_id}/{model_id}
                    output_path = model_path.parent.parent.parent / export_subdir
                else:
                    # single scene run, store it in the model directory
                    output_path = model_path / export_subdir

                export_scenes_predictions(
                    [model_path],
                    output_path,
                    config.lang.model,
                    config.export_scannetpp.knn_k,
                    config.export_scannetpp.sem_topk,
                    config.export_scannetpp.use_segments,
                )
                print("\n\n")

    if wandb.run is not None:
        wandb.finish()
