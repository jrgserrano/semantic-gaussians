import torch
import torch.nn.functional as F

from core.utils.loss_utils import get_mean_prototypes, sample_pixels


def instance_2d_mean_contrastive_loss(
    network_output: torch.Tensor,
    gt_masks: torch.Tensor,
    mask_dim: int,
    sample_size: int = -1,
    gamma: float = 1.0,
    normalize: bool = False,
    weights: tuple[float, float] = (1.0, 1.0),
) -> dict[str, torch.Tensor]:
    """Contrastive Instance loss based on mean prototypes.
    Compute the mean prototype for each instance and use it as the target for the contrastive loss.
    The pixel should be close to the mean prototype of the same instance and the prototypes should be far from the prototypes of other instances.
    """

    assert gt_masks.ndim == 2, f"Expected 2D masks tensor, got {gt_masks.ndim - 1}D"
    assert gt_masks.dtype == torch.long, (
        f"Expected long masks tensor, got {gt_masks.dtype}"
    )
    if normalize:
        network_output = F.normalize(network_output, dim=0)

    assert network_output.shape[0] == mask_dim, (
        "Mask level dimensions and feature dimensions mismatch"
    )

    labels: torch.Tensor = gt_masks.unique()
    if len(labels) == 1 and labels.item() == -1:
        # TODO: filter masks during load time
        return {"total": torch.tensor(0.0, device=network_output.device)}

    sample_features, sample_targets, _ = sample_pixels(
        network_output[:mask_dim], gt_masks, sample_size, ignore_label=-1
    )

    with torch.cuda.nvtx.range("inst2d-loss_mean-prototypes"):
        mean_prototypes = get_mean_prototypes(network_output[:mask_dim], gt_masks)[
            0
        ]  # (I, C)

    feats_pos = sample_features
    mean_pos = mean_prototypes
    mean_neg = mean_prototypes

    # positive contrastive loss
    assert sample_targets.ndim == 1, (
        f"Expected 1D targets tensor, got {sample_targets.ndim}D"
    )

    mean_p = mean_pos.gather(
        0, sample_targets.unsqueeze(1).expand(-1, mean_pos.shape[1])
    )
    loss_pos = (feats_pos - mean_p).pow(2).sum(dim=-1)  # (#samples)

    # negative contrastive loss
    if mean_neg.size(0) > 1:
        loss_neg = torch.cdist(mean_neg, mean_neg)  # (I, I)

        n = loss_neg.size(0)
        triu_indices = torch.triu_indices(
            n,
            n,
            offset=1,
            device=loss_neg.device,
        )

        # # Extract the upper triangle part
        triu_indices = triu_indices[0] * n + triu_indices[1]
        loss_neg = loss_neg.flatten().gather(0, triu_indices)

        # Maximize the distance between prototypes based on a margin given by gamma
        loss_neg = F.relu(gamma - loss_neg)
    else:
        loss_neg = None

    loss = weights[0] * loss_pos.mean()
    if loss_neg is not None:
        loss += weights[1] * loss_neg.mean()

    loss_output = {"positive": loss_pos.mean()}

    if loss_neg is not None:
        loss_output["negative"] = loss_neg.mean()

    loss_output["total"] = loss

    return loss_output


def instance_2d_loss(
    network_output: torch.Tensor,
    gt_masks: torch.Tensor,
    mask_dim: int,
    sample_size: int = -1,
    gamma: float = 1.0,
    weights: list[float] = [1.0, 1.0],
    normalize: bool = False,
) -> dict[str, torch.Tensor]:
    assert len(weights) == 2, "Expected two weights"
    return instance_2d_mean_contrastive_loss(
        network_output,
        gt_masks,
        mask_dim,
        sample_size,
        gamma,
        normalize,
        (weights[0], weights[1]),
    )
