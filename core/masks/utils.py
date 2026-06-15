from typing import Any

import numpy as np
import torch


# LangSplat post-processing utilities
def filter(keep: torch.Tensor, masks_result: list[dict[str, Any]]):
    keep_ = keep.int().cpu().numpy()
    result_keep: list[dict[str, Any]] = []
    for i, m in enumerate(masks_result):
        if i in keep_:
            result_keep.append(m)
    return result_keep


def mask_nms(
    masks: torch.Tensor, scores: torch.Tensor, iou_thr=0.7, score_thr=0.1, inner_thr=0.2
):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.

    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros(
        (num_masks,) * 2, dtype=torch.float, device=masks.device
    )
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(
                torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float
            )
            union = torch.sum(
                torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float
            )
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if (
                intersection / masks_area[i] < 0.5
                and intersection / masks_area[j] >= 0.85
            ):
                inner_iou = 1 - (intersection / masks_area[j]) * (
                    intersection / masks_area[i]
                )
                inner_iou_matrix[i, j] = inner_iou
            if (
                intersection / masks_area[i] >= 0.85
                and intersection / masks_area[j] < 0.5
            ):
                inner_iou = 1 - (intersection / masks_area[j]) * (
                    intersection / masks_area[i]
                )
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(
    masks: list[dict[str, Any]],
    iou_thr: float,
    score_thr: float,
    inner_thr: float,
):
    # remove redundant masks based on the scores and overlap rate between masks
    seg_pred = torch.from_numpy(np.stack([m["segmentation"] for m in masks], axis=0))
    iou_pred = torch.from_numpy(np.stack([m["predicted_iou"] for m in masks], axis=0))
    stability = torch.from_numpy(
        np.stack([m["stability_score"] for m in masks], axis=0)
    )

    scores = stability * iou_pred
    keep_mask_nms = mask_nms(
        seg_pred, scores, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr
    )
    masks = filter(keep_mask_nms, masks)

    return masks
