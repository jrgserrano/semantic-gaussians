import typing

import torch


def masks_encode_binary(masks: torch.Tensor) -> torch.BoolTensor:
    assert masks.ndim == 2, "masks must be 2D"
    assert masks.dtype == torch.long, "only long tensors are supported"
    num_instances = int(
        masks.max().item() + 1
    )  # number of instances without ignore label
    binary_masks = masks + 1  # shift indices so that 0 is ignore label
    binary_masks = torch.nn.functional.one_hot(
        binary_masks,
        num_classes=num_instances + 1,  # +1 for ignore label
    ).bool()
    binary_masks = binary_masks.permute(2, 0, 1).contiguous()  # (I, H, W)
    # binary_masks = binary_masks[1:]  # remove ignore label
    return typing.cast(torch.BoolTensor, binary_masks)


def masks_encode_long(masks: torch.Tensor) -> torch.LongTensor:
    """Convert binary masks to long masks.
    Args:
        masks: binary masks (I, H, W)"""
    assert masks.ndim == 3, "masks must be 3D"
    assert masks.dtype == torch.bool, "only boolean tensors are supported"
    # "argmax_cpu" not implemented for 'Bool'
    output = -1 * torch.ones(masks.shape[1:], device=masks.device, dtype=torch.long)
    for i, mask in enumerate(masks):
        output[mask] = i
    return typing.cast(torch.LongTensor, output)


def mask_consecutive_labels(masks: torch.Tensor) -> torch.Tensor:
    assert masks.ndim == 2, "masks must be 2D"
    assert masks.dtype == torch.long, "only long tensors are supported"
    labels = masks.unique()
    labels = labels[labels >= 0]  # ignore -1 label
    output = -1 * torch.ones_like(masks)
    for i, label in enumerate(labels):
        output[masks == label] = i
    return output
