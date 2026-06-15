import math
from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF

from core.gaussian_model import GaussianModel
from core.language import LanguageModel
from core.masks.sam_levels_model import batched_mask_to_box
from core.params import ModelParams, PipeParams
from core.scene import Camera


@dataclass
class RenderParams:
    gaussians: GaussianModel
    model_params: ModelParams
    cameras: list[Camera]
    pipe_params: PipeParams
    bg: torch.Tensor


@dataclass
class CropParams:
    img_size: int | tuple[int, int]
    levels: int
    masked_crop: bool
    expansion_ratio: float
    dynamic_ratio: bool
    alpha_blend: float


def seg_crop_pad_resize(
    image: torch.Tensor,
    masks: torch.Tensor,
    crop_boxes: torch.Tensor,
    img_size: int | tuple[int, int] = 224,
    masked: bool = False,
    alpha_blend: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert image.dtype == torch.uint8, (
        f"image must be uint8 [0, 255], but got {image.dtype}"
    )
    assert len(masks) == len(crop_boxes)
    assert 0.0 <= alpha_blend <= 1.0, (
        f"alpha_blend must be in [0, 1], but got {alpha_blend}"
    )
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    crops = []
    crop_masks = []
    for i, (mask, box) in enumerate(zip(masks, crop_boxes)):
        x1, y1, x2, y2 = box
        # Boxes contain the pixel coordinates of the bounding box such that the object is perfectly inside the box.
        # To use these coordinates to crop the object, we need to add 1 to the end coordinates.
        crop = image[:, y1 : y2 + 1, x1 : x2 + 1]
        crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1].unsqueeze(0)
        if masked:
            crop = crop * crop_mask + alpha_blend * crop * ~crop_mask
            crop = crop.to(image.dtype)

        max_dim = max(crop.shape[1:])
        pt = (max_dim - crop.shape[1]) // 2
        pb = math.ceil((max_dim - crop.shape[1]) / 2)
        pl = (max_dim - crop.shape[2]) // 2
        pr = math.ceil((max_dim - crop.shape[2]) / 2)
        crop = TF.pad(crop, [pl, pt, pr, pb])
        crop_mask = TF.pad(crop_mask, [pl, pt, pr, pb])
        crop = TF.resize(crop, list(img_size))
        crop_mask = TF.resize(
            crop_mask, list(img_size), interpolation=TF.InterpolationMode.NEAREST
        )
        crops.append(crop)
        crop_masks.append(crop_mask)

    assert len(crops), f"#masks: {len(masks)}"
    crops = torch.stack(crops)
    crop_masks = torch.stack(crop_masks)
    assert crops.shape[-2:] == img_size
    return crops, crop_masks


def seg_pad_resize_masq(
    image: torch.Tensor, masks: torch.Tensor, img_size: int | tuple[int, int] = 336
):
    assert image.dtype == torch.uint8, (
        f"image must be uint8 [0, 255], but got {image.dtype}"
    )
    if isinstance(img_size, tuple):
        assert img_size[0] == img_size[1], "img_size must be square"
        img_size = img_size[0]

    crops = TF.resize(
        image.float().div(255.0), [img_size - 1], max_size=img_size
    ).unsqueeze(0)
    crop_masks = masks.unsqueeze(0).float()  # add batch dimension
    crop_masks = TF.resize(
        crop_masks,
        [img_size - 1],
        interpolation=TF.InterpolationMode.NEAREST,
        max_size=img_size,
    )

    crops = TF.center_crop(crops, [img_size])
    crop_masks = TF.center_crop(crop_masks, [img_size])
    assert crops.shape[-2:] == (img_size, img_size)
    assert crops.shape[-2:] == crop_masks.shape[-2:]
    return crops.mul(255.0).to(torch.uint8), crop_masks.bool()


def multi_level_masks_to_boxes(
    masks: torch.Tensor, levels: int, expansion_ratio: float
):
    # copied and modified from https://github.com/OpenMask3D/openmask3d/blob/3bc3fc52693b25668d0e91d55a2ea714544a4749/openmask3d/mask_features_computation/utils.py#L22
    assert masks.ndim == 3, f"masks must be 3D (B, H, W), but got {masks.ndim}"
    all_crop_boxes: list[torch.Tensor] = []
    all_obj_boxes: list[torch.Tensor] = []
    for mask in masks:
        shape = mask.shape
        box = batched_mask_to_box(mask.unsqueeze(0)).squeeze(0)
        crop_boxes: list[torch.Tensor] = [box]
        obj_boxes: list[torch.Tensor] = [
            box - torch.tensor([box[0], box[1], box[0], box[1]])
        ]
        x_exp = ((box[2] - box[0]).abs() * expansion_ratio).int()
        y_exp = ((box[3] - box[1]).abs() * expansion_ratio).int()
        for i in range(1, levels):
            level = i + 1
            x_exp_lvl = x_exp * level
            y_exp_lvl = y_exp * level
            crop_box = torch.tensor(
                [
                    max(0, (box[0] - x_exp_lvl).item()),
                    max(0, (box[1] - y_exp_lvl).item()),
                    min(shape[1], (box[2] + x_exp_lvl).item()),
                    min(shape[0], (box[3] + y_exp_lvl).item()),
                ]
            ).int()
            crop_boxes.append(crop_box)
            obj_box = torch.tensor(
                [
                    box[0] - crop_box[0],
                    box[1] - crop_box[1],
                    box[2] - crop_box[0],
                    box[3] - crop_box[1],
                ]
            ).int()
            obj_boxes.append(obj_box)
        all_crop_boxes.append(torch.stack(crop_boxes))
        all_obj_boxes.append(torch.stack(obj_boxes))
    return torch.stack(all_crop_boxes), torch.stack(all_obj_boxes)


def masks_to_crops(
    image: torch.Tensor,
    masks: torch.Tensor,
    crop_params: CropParams,
    lang_model: LanguageModel | None,
    crop_boxes: torch.Tensor | None = None,
    obj_boxes: torch.Tensor | None = None,
):
    """
    Create crops from masks based on multi-level boxes and the provided crop params.
    If lang_model is not None, preprocess the crops for the language model.
    """
    # masks = (M, H, W)
    assert (crop_boxes is None) == (obj_boxes is None), (
        "crop_boxes and obj_boxes must be both None or both not None"
    )
    if crop_boxes is not None and obj_boxes is not None:
        assert crop_boxes.shape == obj_boxes.shape, (
            "crop_boxes and obj_boxes must match"
        )
        assert crop_boxes.ndim == 3, "crop_boxes and obj_boxes must be 3D (M, L, 4)"
        assert crop_boxes.size(0) == masks.size(0), (
            "Masks and crop_boxes and obj_boxes must match"
        )
    else:
        crop_boxes, obj_boxes = multi_level_masks_to_boxes(
            masks, crop_params.levels, crop_params.expansion_ratio
        )  # (M, L, 4), (M, L, 4)
    crops, crop_masks = seg_crop_pad_resize(
        image,
        masks.repeat(crop_boxes.size(1), 1, 1),
        crop_boxes.view(-1, 4),
        crop_params.img_size,
        crop_params.masked_crop,
        crop_params.alpha_blend,
    )  # (M * L, C, CH, CW), (M * L, CH, CW)
    if lang_model is not None:
        assert lang_model.img_size == crop_params.img_size
        crops = crops.permute(0, 2, 3, 1).contiguous()  # (M * L, CH, CW, C)
        crops = lang_model.preprocess_images(crops.numpy())  # (M * L, C, CH, CW)
    crops = crops.view(*crop_boxes.shape[:2], *crops.shape[-3:])  # (M, L, C, CH, CW)
    crop_masks = crop_masks.squeeze(-3).view(
        *crop_boxes.shape[:2], *crop_masks.shape[-2:]
    )  # (M, L, CH, CW)
    return crops, crop_masks, crop_boxes, obj_boxes
