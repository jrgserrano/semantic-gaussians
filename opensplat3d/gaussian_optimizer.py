import torch
from torch import nn

from opensplat3d.gaussian_model import GaussianModel
from opensplat3d.params import OptimizationParams
from opensplat3d.utils.general_utils import (
    build_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
)


def replace_tensor_to_optimizer(
    optimizer: torch.optim.Optimizer, tensor: torch.Tensor, name: str
):
    optimizable_tensors: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        if group["name"] == name:
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor)
            if stored_state is not None:
                optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def cat_tensors_to_optimizer(
    optimizer: torch.optim.Optimizer, tensors_dict: dict[str, torch.Tensor]
):
    optimizable_tensors: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1
        extension_tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                dim=0,
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0)
            )
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0)
            )
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def prune_optimizer(optimizer: torch.optim.Optimizer, mask: torch.BoolTensor):
    optimizable_tensors: dict[str, torch.Tensor] = {}
    for group in optimizer.param_groups:
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(group["params"][0][mask])
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(group["params"][0][mask])
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


class GaussianOptimizer:
    def __init__(
        self,
        model: GaussianModel,
        opt_params: OptimizationParams,
        sh_degree: int,
        spatial_lr_scale: float,
        static_xyz: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        # TODO add better support for freezing all parameters except features
        # Exclude them early before creating the optimizer

        self.model = model
        self.percent_dense = opt_params.percent_dense
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.spatial_lr_scale = spatial_lr_scale
        self.static_xyz = static_xyz

        self.max_radii2D = torch.zeros((self.model.num_points), device=device)
        self.xyz_gradient_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.importance_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.denom = torch.zeros((self.model.num_points, 1), device=device)
        self.feature_instability_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.rgb_error_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.semantic_error_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.fim_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.feature_delta_accum = 0.0
        self.feature_delta_denom = 0

        params = []
        if not static_xyz:
            params.append(
                {
                    "params": [self.model._xyz],
                    "lr": opt_params.position_lr_init * spatial_lr_scale,
                    "name": "xyz",
                }
            )

        params += [
            {
                "params": [self.model._features_dc],
                "lr": opt_params.sh_lr,
                "name": "f_dc",
            },
            {
                "params": [self.model._features_rest],
                "lr": opt_params.sh_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self.model._opacity],
                "lr": opt_params.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self.model._scaling],
                "lr": opt_params.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self.model._rotation],
                "lr": opt_params.rotation_lr,
                "name": "rotation",
            },
        ]
        if self.model._features is not None:
            params.append(
                {
                    "params": [self.model._features],
                    "lr": opt_params.feature_lr,
                    "name": "features",
                }
            )

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            params, lr=0.0, eps=1e-15
        )
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=opt_params.position_lr_init * spatial_lr_scale,
            lr_final=opt_params.position_lr_final * spatial_lr_scale,
            lr_delay_mult=opt_params.position_lr_delay_mult,
            max_steps=opt_params.position_lr_max_steps,
        )

    def to(self, device: torch.device):
        self.max_radii2D = self.max_radii2D.to(device)
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
        self.importance_accum = self.importance_accum.to(device)
        self.denom = self.denom.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    def cpu(self):
        return self.to(torch.device("cpu"))

    def state_dict(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "importance_accum": self.importance_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
        }

    def load_state_dict(self, state_dict: dict):
        self.active_sh_degree = state_dict["active_sh_degree"]
        self.max_radii2D = state_dict["max_radii2D"]
        self.xyz_gradient_accum = state_dict.get("xyz_gradient_accum", torch.zeros_like(self.xyz_gradient_accum))
        self.importance_accum = state_dict.get("importance_accum", torch.zeros_like(self.importance_accum))
        self.denom = state_dict["denom"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.spatial_lr_scale = state_dict["spatial_lr_scale"]

    def freeze(self):
        self.model._xyz.requires_grad = False
        self.model._rotation.requires_grad = False
        self.model._scaling.requires_grad = False
        self.model._opacity.requires_grad = False
        self.model._features_dc.requires_grad = False
        self.model._features_rest.requires_grad = False

    def unfreeze(self):
        self.model._xyz.requires_grad = True
        self.model._rotation.requires_grad = True
        self.model._scaling.requires_grad = True
        self.model._opacity.requires_grad = True
        self.model._features_dc.requires_grad = True
        self.model._features_rest.requires_grad = True

    def update_learning_rate(self, iteration: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def oneup_sh_degree(self):
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(
                self.model.get_opacity, torch.ones_like(self.model.get_opacity) * 0.01
            )
        )
        optimizable_tensors = replace_tensor_to_optimizer(
            self.optimizer, opacities_new, "opacity"
        )
        self.model._opacity = optimizable_tensors["opacity"]

    def add_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        visibility_filter: torch.Tensor,
        radii: torch.Tensor,
        xyz_grad_rgb: torch.Tensor | None = None,
        xyz_grad_sem: torch.Tensor | None = None,
    ):
        # Keep track of max radii in image-space for pruning
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.xyz_gradient_accum[visibility_filter] += torch.norm(
            viewspace_point_tensor.grad[visibility_filter, :2],  # type: ignore
            dim=-1,
            keepdim=True,
        )
        
        if xyz_grad_rgb is not None:
             self.rgb_error_accum[visibility_filter] += torch.norm(
                xyz_grad_rgb[visibility_filter, :2],
                dim=-1,
                keepdim=True,
            )
        
        if xyz_grad_sem is not None:
             self.semantic_error_accum[visibility_filter] += torch.norm(
                xyz_grad_sem[visibility_filter, :2],
                dim=-1,
                keepdim=True,
            )

        # Accumulate Fisher Information (squared gradients)
        self.fim_accum[visibility_filter] += (viewspace_point_tensor.grad[visibility_filter, :2] ** 2).sum(dim=-1, keepdim=True)

        self.denom[visibility_filter] += 1

    def add_importance_stats(self, opacity_grad_rgb: torch.Tensor | None = None, opacity_grad_feat: torch.Tensor | None = None, opt: OptimizationParams | None = None):
        """
        Accumulates the weighted importance score psi_i (FeatureSLAM).
        psi_i = sum( lambda_c * |d_alpha/d_Lrgb| + lambda_f * |d_alpha/d_Lfeat| )
        """
        if opacity_grad_rgb is not None and opacity_grad_feat is not None and opt is not None:
            # Weighted importance
            score = opt.lambda_c * torch.abs(opacity_grad_rgb) + opt.lambda_f * torch.abs(opacity_grad_feat)
            self.importance_accum += score
        elif self.model._opacity.grad is not None:
            # Fallback to total gradient if components aren't provided
            self.importance_accum += torch.abs(self.model._opacity.grad)

    @torch.no_grad()
    def record_feature_delta(self, old_features: torch.Tensor, new_features: torch.Tensor):
        """
        Tracks the mean absolute change of Gaussian features between optimization steps.
        Used as a diagnostic for convergence in semantic/instance space.
        """
        diff = (new_features - old_features).abs().mean(dim=-1, keepdim=True)
        self.feature_instability_accum += diff
        
        delta = diff.mean().item()
        self.feature_delta_accum += delta
        self.feature_delta_denom += 1

    def update_model(self, optimizable_tensors: dict[str, torch.Tensor]):
        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]
        if "features" in optimizable_tensors:
            self.model._features = optimizable_tensors["features"]

    def prune_points(self, mask: torch.BoolTensor):
        valid_points_mask: torch.BoolTensor = ~mask  # type: ignore
        optimizable_tensors = prune_optimizer(self.optimizer, valid_points_mask)

        # update model
        self.update_model(optimizable_tensors)

        # update densification state
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.importance_accum = self.importance_accum[valid_points_mask]
        self.feature_instability_accum = self.feature_instability_accum[valid_points_mask]
        self.rgb_error_accum = self.rgb_error_accum[valid_points_mask]
        self.semantic_error_accum = self.semantic_error_accum[valid_points_mask]
        self.fim_accum = self.fim_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_features: torch.Tensor | None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if new_features is not None:
            d["features"] = new_features

        optimizable_tensors = cat_tensors_to_optimizer(self.optimizer, d)

        # update model
        self.update_model(optimizable_tensors)

        # reset densification state
        device = self.denom.device
        self.max_radii2D = torch.zeros((self.model.num_points), device=device)
        self.xyz_gradient_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.importance_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.feature_instability_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.rgb_error_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.semantic_error_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.fim_accum = torch.zeros((self.model.num_points, 1), device=device)
        self.denom = torch.zeros((self.model.num_points, 1), device=device)

    def densify_and_clone(
        self, grads: torch.Tensor, grad_threshold: float, scene_extent: float
    ):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self.model._xyz[selected_pts_mask]
        new_features_dc = self.model._features_dc[selected_pts_mask]
        new_features_rest = self.model._features_rest[selected_pts_mask]
        new_opacities = self.model._opacity[selected_pts_mask]
        new_scaling = self.model._scaling[selected_pts_mask]
        new_rotation = self.model._rotation[selected_pts_mask]

        new_features = (
            self.model._features[selected_pts_mask]
            if self.model._features is not None
            else None
        )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_features,
        )

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
    ):
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((self.model.num_points), device=grads.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask: torch.BoolTensor = torch.where(  # type: ignore
            padded_grad >= grad_threshold, True, False
        )
        selected_pts_mask: torch.BoolTensor = torch.logical_and(  # type: ignore
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.model.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=stds.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(
            -1
        ) + self.model.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.model.scaling_inverse_activation(
            self.model.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self.model._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.model._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.model._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self.model._opacity[selected_pts_mask].repeat(N, 1)
        new_features = (
            self.model._features[selected_pts_mask].repeat(N, 1)
            if self.model._features is not None
            else None
        )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_features,
        )

        prune_filter: torch.BoolTensor = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),  # type: ignore
                    device=selected_pts_mask.device,
                    dtype=torch.bool,
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: float | int | None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask: torch.BoolTensor = (self.model.get_opacity < min_opacity).squeeze()  # type: ignore
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.model.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(  # type: ignore
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        num_pruned = prune_mask.sum().item()
        self.prune_points(prune_mask)
        return {"pruned": num_pruned, "mean_metric": 0.0}

    def semantic_pruning(self, percentile: float) -> dict:
        """
        Prunes Gaussians with an importance score below the given percentile.
        Guarantees that only 'meaningful' Gaussians (high-grad) are kept.
        """
        if self.denom.sum() == 0 or percentile <= 0 or percentile >= 1:
            return {"pruned": 0, "mean_metric": 0.0}

        # Importance score normalized by views seen
        score = self.importance_accum / (self.denom + 1e-8)
        threshold = torch.quantile(score, percentile)
        prune_mask = (score <= threshold).squeeze()
        
        num_pruned = prune_mask.sum().item()
        mean_val = score[prune_mask].mean().item() if num_pruned > 0 else 0.0
        
        if num_pruned > 0:
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        return {"pruned": num_pruned, "mean_metric": mean_val}

    @torch.no_grad()
    def redundancy_pruning(self, tau_dist: float, tau_sim: float, K: int = 4) -> dict:
        """
        LEGO-SLAM redundancy pruning.
        Checks if a Gaussian is too close and too semantically similar to its neighbors.
        """
        xyz = self.model.get_xyz
        features = self.model.get_features
        if features is None:
            return {"pruned": 0, "mean_metric": 0.0}

        # Normalize features for cosine similarity
        features = torch.nn.functional.normalize(features, dim=1)
        
        # Memory-efficient redundancy check (safe for 8GB VRAM)
        num_pts = xyz.shape[0]
        block_size = 2000 # Smaller blocks for 8GB safety
        prune_mask = torch.zeros(num_pts, dtype=torch.bool, device=xyz.device)
        
        # We sample a subset of neighbors to check against to avoid OOM
        # For a truly robust N-logN search, we'd need a KD-Tree, but here 
        # we'll use a chunked batch approach.
        for i in range(0, num_pts, block_size):
            end = min(i + block_size, num_pts)
            # Compare block against itself + a random sample of the cloud
            # to keep memory low while catching global redundancy
            sample_idx = torch.randint(0, num_pts, (block_size,), device=xyz.device)
            compare_xyz = torch.cat([xyz[i:end], xyz[sample_idx]], dim=0)
            compare_feat = torch.cat([features[i:end], features[sample_idx]], dim=0)
            
            dists = torch.cdist(xyz[i:end], compare_xyz)
            # Mask out self-comparison (diagonal)
            dists[:, : (end - i)].fill_diagonal_(float("inf"))
            
            near_vals, near_idx = dists.topk(K, largest=False)
            spat_mask = (near_vals < tau_dist).any(dim=1)
            
            sim_mask = torch.zeros_like(spat_mask)
            for k in range(K):
                sim = (features[i:end] * compare_feat[near_idx[:, k]]).sum(dim=1)
                sim_mask = torch.logical_or(sim_mask, sim > tau_sim)
            
            prune_mask[i:end] = torch.logical_and(spat_mask, sim_mask)

        num_pruned = prune_mask.sum().item()
        if num_pruned > 0:
            self.prune_points(prune_mask)
            
        return {"pruned": num_pruned, "mean_metric": tau_dist}

    @torch.no_grad()
    def scale_guided_pruning(self, theta_scale: float, theta_ratio: float) -> dict:
        """
        OpenGS-SLAM boundary and anisotropy pruning.
        Removes Gaussians that are excessively large or excessively thin/elongated.
        """
        scales = self.model.get_scaling
        max_scales = torch.max(scales, dim=1).values
        min_scales = torch.min(scales, dim=1).values
        
        # 1. Absolute size check
        large_mask = max_scales > theta_scale
        
        # 2. Anisotropy (ratio) check
        ratio_mask = (max_scales / (min_scales + 1e-7)) > theta_ratio
        
        prune_mask = torch.logical_or(large_mask, ratio_mask)
        
        num_pruned = prune_mask.sum().item()
        mean_val = max_scales[prune_mask].mean().item() if num_pruned > 0 else 0.0
        
        if num_pruned > 0:
            self.prune_points(prune_mask)
            
        return {"pruned": num_pruned, "mean_metric": mean_val}

    @torch.no_grad()
    def statistical_outlier_removal_pruning(self, K: int, std_ratio: float) -> dict:
        """
        Removes 'floaters' by checking the average distance to K-nearest neighbors.
        If a point's mean distance is > (global_mean + std_ratio * global_std), it is pruned.
        """
        xyz = self.model.get_xyz
        num_pts = xyz.shape[0]
        if num_pts <= K:
            return {"pruned": 0, "mean_metric": 0.0}

        block_size = 128
        mean_dists = torch.zeros(num_pts, device=xyz.device)

        # Chunked KNN search to stay within VRAM limits
        for i in range(0, num_pts, block_size):
            end = min(i + block_size, num_pts)
            # Find nearest neighbors across the entire cloud
            dists = torch.cdist(xyz[i:end], xyz)
            # Exclude self (distance 0)
            dists.fill_diagonal_(float("inf"))
            
            vals, _ = dists.topk(K, largest=False)
            mean_dists[i:end] = vals.mean(dim=1)
            
            # Help PyTorch reclaim memory from the large dists tensor
            del dists
            if i % 1024 == 0:
                torch.cuda.empty_cache()

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        threshold = global_mean + std_ratio * global_std

        prune_mask = mean_dists > threshold
        num_pruned = prune_mask.sum().item()

        if num_pruned > 0:
            self.prune_points(prune_mask)

        return {"pruned": num_pruned, "mean_metric": threshold.item()}

    def densify_and_split_by_instability(self, instability_weight: float, scene_extent: float):
        """
        Splits Gaussians that are semantically 'unstable' (frequently changing identity).
        """
        # Average instability per iteration
        instability = self.feature_instability_accum / (self.denom + 1e-7)
        
        # We target the most unstable points (top 5% or based on weight)
        # Weight scales the threshold relative to the mean instability
        threshold = instability.mean() * (1.0 / (instability_weight + 1e-7))
        
        selected_pts_mask = (instability > threshold).reshape(-1)
        # Further filter by size to prevent splitting tiny points
        selected_pts_mask &= (torch.max(self.model.get_scaling, dim=1).values > 0.01 * scene_extent)
        
        if selected_pts_mask.any():
            stds = self.model.get_scaling[selected_pts_mask]
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self.model._rotation[selected_pts_mask])
            
            new_xyz = (torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + 
                       self.model.get_xyz[selected_pts_mask])
            
            # Decrease scaling for both the original and the new point
            new_scaling = self.model.get_scaling[selected_pts_mask] / 1.6
            self.model._scaling[selected_pts_mask] = torch.log(new_scaling)
            self.model._rotation[selected_pts_mask] = self.model._rotation[selected_pts_mask]
            
            # Carry over features and other attributes
            self.densification_postfix(
                new_xyz, 
                self.model._features_dc[selected_pts_mask], 
                self.model._features_rest[selected_pts_mask],
                self.model._opacity[selected_pts_mask],
                torch.log(new_scaling),
                self.model._rotation[selected_pts_mask],
                self.model._features[selected_pts_mask] if self.model._features is not None else None
            )
            
            return selected_pts_mask.sum().item()
        return 0

    def smart_refine(
        self, 
        iteration: int, 
        opt: OptimizationParams, 
        scene_extent: float,
        w_rgb: float = 1.0,
        w_sem: float = 2.0,
        w_inst: float = 1.0
    ):
        """
        Uses accumulated error diagnostics to perform prioritized densification and pruning.
        """
        device = self.denom.device
        # 1. Calculate Combined Importance Score
        # Normalize by denom (number of views seen)
        valid_mask = (self.denom > 0).squeeze()
        
        avg_rgb = self.rgb_error_accum / (self.denom + 1e-7)
        avg_sem = self.semantic_error_accum / (self.denom + 1e-7)
        avg_inst = self.feature_instability_accum / (self.denom + 1e-7)
        
        # Combined score for densification
        combined_score = (w_rgb * avg_rgb + w_sem * avg_sem + w_inst * avg_inst)
        
        # 2. Smart Densification (Splitting)
        # We target a percentile of the most 'problematic' points
        if iteration < opt.densify_until_iter:
            # Combined score threshold based on distribution
            threshold = combined_score[valid_mask].mean() * 1.5
            self.densify_and_split(combined_score, threshold, scene_extent)

        # 3. Smart Pruning (Targeting Semantic Floaters)
        # Gaussians with high semantic error but very low RGB gradient contribution
        # are likely floaters that mismatch the segments.
        if iteration > opt.densify_from_iter:
            # High semantic error AND low RGB influence
            # We use 2x mean as a simple outlier detector
            semantic_outlier_mask = (avg_sem > (2.0 * avg_sem[valid_mask].mean()))
            low_rgb_mask = (avg_rgb < (0.5 * avg_rgb[valid_mask].mean()))
            
            prune_mask = (semantic_outlier_mask & low_rgb_mask).squeeze()
            num_pruned = prune_mask.sum().item()
            if num_pruned > 0:
                self.prune_points(prune_mask)
                print(f"[ SMART ] Pruned {num_pruned} semantic floaters at iter {iteration}")

        return combined_score
