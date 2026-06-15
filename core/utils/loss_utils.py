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

from math import exp

import torch
from torch.autograd import Variable

from core.utils.mask_utils import masks_encode_binary


def gaussian(window_size: int, sigma: float):
    gauss = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


@torch.cuda.nvtx.range("sample_pixels")
def sample_pixels(
    network_output: torch.Tensor,
    gt: torch.Tensor,
    sample_size: int = 4096,
    ignore_label: int | None = None,
):
    """Sample pixels from the network output and ground truth.
    Returns the sampled features, targets and indices.
    gt: (..., H, W) int tensor with values in [-1, I-1] where L is the number of mask levels, I is the number of instances and -1 is an ignore label.
    """
    assert gt.dtype == torch.long, f"Expected long masks tensor, got {gt.dtype}"

    gt_flat = gt.flatten(-2, -1)
    gt_ignore = None
    if ignore_label is not None:
        gt_ignore = gt_flat == ignore_label
        if gt_ignore.ndim == 2:
            # binary_or operation over mask levels
            gt_ignore = gt_ignore.sum(dim=0).bool()
        gt_flat = gt_flat[..., ~gt_ignore]

    sample_indices = None
    sample_targets = gt_flat
    if sample_size > 0:
        sample_indices = torch.randperm(gt_flat.size(-1), device=gt_flat.device)[
            :sample_size
        ]
        sample_targets = gt_flat.gather(-1, sample_indices)

    network_output_flat = network_output.flatten(1).permute(1, 0).contiguous()
    if gt_ignore is not None:
        network_output_flat = network_output_flat[~gt_ignore]

    sample_features = network_output_flat
    if sample_indices is not None:
        # sample_features = network_output_flat[sample_indices]
        sample_features = network_output_flat.gather(
            0, sample_indices.unsqueeze(-1).expand(-1, network_output_flat.shape[1])
        )

    return sample_features, sample_targets, sample_indices


def get_mean_prototypes(
    network_output: torch.Tensor,
    gt: torch.Tensor,
):
    """Compute the mean prototype for each instance.
    Returns:
    - mean_prototypes: (I, C) tensor with the mean prototype for each instance
    - binary_gt: (I, H * W) tensor with the binary masks
    - Np: (I) tensor with the number of pixels for each instance
    """
    # Convert masks to binary masks

    with torch.cuda.nvtx.range("mean-prototypes-binary"):
        binary_gt = masks_encode_binary(gt)
    binary_gt = binary_gt[1:]  # remove ignore label
    binary_gt = binary_gt.flatten(1)  # (I, H * W)

    network_output = network_output.flatten(1)  # (C, H * W)

    with torch.cuda.nvtx.range("mean-prototypes-sum"):
        Np = binary_gt.sum(dim=-1)
        assert Np.min() > 0, "Some instances have no pixels"
    with torch.cuda.nvtx.range("mean-prototypes-mult"):
        cluster_features = network_output.unsqueeze(0) * binary_gt.unsqueeze(
            1
        )  # (I, C, H * W)
    with torch.cuda.nvtx.range("mean-prototypes-div"):
        mean_prototypes = cluster_features.sum(dim=-1) / Np.unsqueeze(-1)  # (I, C)

    return mean_prototypes, binary_gt, Np


def pearson_correlation_loss(pred, target, valid_mask=None):
    if valid_mask is not None:
        p = pred[valid_mask]
        t = target[valid_mask]
    else:
        p = pred.flatten()
        t = target.flatten()
        
    if p.numel() < 2:
        return torch.tensor(0.0, device=pred.device)
        
    p_mean = p.mean()
    t_mean = t.mean()
    p_centered = p - p_mean
    t_centered = t - t_mean
    
    cov = (p_centered * t_centered).sum()
    p_var = (p_centered ** 2).sum()
    t_var = (t_centered ** 2).sum()
    
    denom = torch.sqrt(p_var * t_var)
    if denom < 1e-6:
        return torch.tensor(0.0, device=pred.device)
        
    corr = cov / denom
    # Maximize correlation -> minimize 1 - corr
    return 1.0 - corr


def depth_normal_consistency_loss(pred_depth, gt_depth, valid_mask=None):
    if pred_depth.ndim == 2:
        pred_depth = pred_depth.unsqueeze(0)
    if gt_depth.ndim == 2:
        gt_depth = gt_depth.unsqueeze(0)
        
    dx_pred = pred_depth[:, :, 1:] - pred_depth[:, :, :-1]
    dy_pred = pred_depth[:, 1:, :] - pred_depth[:, :-1, :]
    
    dx_gt = gt_depth[:, :, 1:] - gt_depth[:, :, :-1]
    dy_gt = gt_depth[:, 1:, :] - gt_depth[:, :-1, :]
    
    min_h = min(dy_pred.shape[1], dy_gt.shape[1])
    min_w = min(dx_pred.shape[2], dx_gt.shape[2])
    
    dx_pred = dx_pred[:, :min_h, :min_w]
    dy_pred = dy_pred[:, :min_h, :min_w]
    dx_gt = dx_gt[:, :min_h, :min_w]
    dy_gt = dy_gt[:, :min_h, :min_w]
    
    loss_x = torch.abs(dx_pred - dx_gt)
    loss_y = torch.abs(dy_pred - dy_gt)
    
    if valid_mask is not None:
        valid_mask_resized = valid_mask[:min_h, :min_w]
        # ensure there are valid pixels
        if valid_mask_resized.sum() == 0:
            return torch.tensor(0.0, device=pred_depth.device)
        loss = (loss_x[0, valid_mask_resized].mean() + loss_y[0, valid_mask_resized].mean()) / 2.0
    else:
        loss = (loss_x.mean() + loss_y.mean()) / 2.0
        
    if torch.isnan(loss):
        return torch.tensor(0.0, device=pred_depth.device)
    return loss
