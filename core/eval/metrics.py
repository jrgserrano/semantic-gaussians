import numpy as np
import numpy.typing as npt

from core.eval.utils import mask_to_boundary


# Copied from https://github.com/lkeab/gaussian-grouping/blob/main/script/eval_lerf_mask.py#L66C1-L73C15
def calculate_iou(mask1: npt.NDArray, mask2: npt.NDArray) -> float:
    """Calculate IoU between two boolean masks."""
    assert mask1.dtype == bool and mask2.dtype == bool, "Masks must be boolean"
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def calculate_biou(gt: npt.NDArray, dt: npt.NDArray, dilation_ratio: float = 0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, bool): binary mask
    :param dt (numpy array, bool): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = dt.astype("uint8")
    gt = gt.astype("uint8")

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def calculate_loc(bboxes: npt.NDArray, mask: npt.NDArray):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox.astype(int)
        assert x_min < x_max and y_min < y_max, f"Invalid bbox: {bbox}"
        mask_cropped = mask[y_min:y_max, x_min:x_max]
        return bool(mask_cropped.any())
    return False
