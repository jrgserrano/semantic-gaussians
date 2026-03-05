"""
Eval for LERF-Mask with GroundingDINO and SAM.
To get the instances, the mask needs to be clustered based on the features.
"""

import json
import typing
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import tabulate
import torch
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from omegaconf import OmegaConf
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from opensplat3d.config import Config
from opensplat3d.eval.grounding_sam import grounded_sam_output, load_model_hf
from opensplat3d.eval.metrics import (
    calculate_biou,
    calculate_iou,
)
from opensplat3d.eval.utils import (
    generate_eval_output_name,
    get_scene_model_configs,
    render_frames,
)
from opensplat3d.gaussian_model import GaussianModel
from opensplat3d.params import ModelParams
from opensplat3d.scene import Camera
from opensplat3d.utils.mask_utils import masks_encode_binary
from opensplat3d.utils.setup_utils import get_latest_model, setup_from_config


@dataclass
class LERFMaskSceneConfig:
    prompts: list[str]


LERF_MASK_SCENE_CONFIGS = {
    "ramen": LERFMaskSceneConfig(
        prompts=[
            "chopsticks",
            "egg",
            "glass of water",
            "pork belly",
            "wavy noodles in bowl",
            "yellow bowl",
        ]
    ),
    "figurines": LERFMaskSceneConfig(
        prompts=[
            "green apple",
            "green toy chair",
            "old camera",
            "porcelain hand",
            "red apple",
            "red toy chair",
            "rubber duck with red hat",
        ]
    ),
    "teatime": LERFMaskSceneConfig(
        prompts=[
            "apple",
            "bag of cookies",
            "coffee mug",
            "cookies on a plate",
            "paper napkin",
            "plate",
            "sheep",
            "spoon handle",
            "stuffed bear",
            "tea in a glass",
        ]
    ),
}


def lerf_mask_metrics_table_str(metrics: dict[str, Any]):
    table = [("Category", "IoU", "BIoU")]
    table.extend(
        [
            (
                cat_id,
                f"{metrics['miou_per_class'][cat_id] * 100:.2f}",
                f"{metrics['mbiou_per_class'][cat_id] * 100:.2f}",
            )
            for cat_id in metrics["miou_per_class"]
        ]
    )
    table.append(
        (
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
        )
    )
    table.append(
        (
            "Mean",
            f"{metrics['miou'] * 100:.2f}",
            f"{metrics['mbiou'] * 100:.2f}",
        )
    )
    table_str = tabulate.tabulate(table, headers="firstrow", floatfmt=".2f")
    return table_str


def save_lerf_mask_results(
    results: list[tuple[Path, Config, dict]],
    output_path: Path,
    save_txt: bool = True,
    save_json: bool = True,
    print_summary: bool = True,
):
    output_json = {"scenes": {}, "all": {}}
    log_lines: list[str] = []
    all_metrics = {"miou": 0.0, "mbiou": 0.0}
    table = [("Scene", "mIoU", "mBIoU")]

    for model_path, _, metrics in results:
        log_lines.extend(["\n"])
        # model_path = Path(config.model.model_path)
        model_name = model_path.relative_to(model_path.parent.parent)
        log_lines.extend([f"{model_name}\n"])
        table_str = lerf_mask_metrics_table_str(metrics)
        log_lines.extend([table_str, "\n"])
        scene_name = model_path.parent.name
        all_metrics["miou"] += metrics["miou"]
        all_metrics["mbiou"] += metrics["mbiou"]
        table.append(
            (
                str(scene_name),
                f"{metrics['miou'] * 100:.2f}",
                f"{metrics['mbiou'] * 100:.2f}",
            )
        )
        output_json["scenes"][scene_name] = metrics

    if len(results) > 0:
        all_metrics["miou"] /= len(results)
        all_metrics["mbiou"] /= len(results)

    table.append(
        (
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
        )
    )
    table.append(
        (
            "Mean",
            f"{all_metrics['miou'] * 100:.2f}",
            f"{all_metrics['mbiou'] * 100:.2f}",
        )
    )
    output_json["all"] = all_metrics

    log_lines.extend(["\n\n--- Summary ---\n"])
    table_str = tabulate.tabulate(table, headers="firstrow", floatfmt=".2f")
    log_lines.extend([table_str])

    if save_txt:
        with open(output_path / "results.txt", "w") as f:
            f.writelines(log_lines)

    if save_json:
        with open(output_path / "results.json", "w") as f_json:
            json.dump(output_json, f_json)

    if print_summary:
        print("\n--- Summary ---")
        print(table_str)


def calculate_ioa(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate IoA between two boolean masks.
    Args:
        x, y: Boolean masks of shape (B, H, W).
    """
    assert x.dtype == torch.bool and y.dtype == torch.bool, "Masks must be boolean"
    intersection = torch.logical_and(x, y).sum(dim=(-2, -1)).float()
    area = x.sum(dim=(-2, -1))
    intersection[area == 0] = 0
    intersection[area != 0] /= area[area != 0]
    return intersection


@torch.no_grad()
def select_objects_by_ioa(
    obj_masks: torch.Tensor, grounded_masks: list[torch.Tensor], threshold: float = 0.7
):
    """Select objects based on IoA between object masks and grounded masks.
    Args:
        obj_masks: Tensor of shape (H, W).
        grounded_masks: List of tensors of shape (H, W). List has length equal to number of prompts.
        threshold: IoA threshold for object selection.
    """
    results: list[torch.Tensor] = []
    result_ioas: list[torch.Tensor] = []
    assert obj_masks.ndim in {2, 3}, "Object masks must be 2D or 3D"
    if obj_masks.ndim == 2:
        binary_obj_mask = masks_encode_binary(obj_masks)
        binary_obj_mask = binary_obj_mask[1:]  # remove ignore label
    else:
        binary_obj_mask = obj_masks
    assert binary_obj_mask.dtype == torch.bool, (
        f"Object masks must be boolean got ({binary_obj_mask.dtype})"
    )
    for gmask in grounded_masks:
        ioas = calculate_ioa(
            binary_obj_mask, gmask.unsqueeze(0).repeat(binary_obj_mask.shape[0], 1, 1)
        )
        object_ids = (ioas > threshold).nonzero(as_tuple=True)[0]
        results.append(object_ids)
        result_ioas.append(ioas[object_ids])
    return results, result_ioas


def setup_grounded_sam(checkpoint: str | None = None):
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    grounding_dino = typing.cast(
        GroundingDINO, load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
    )
    grounding_dino.cuda()
    grounding_dino.eval()

    sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth" if checkpoint is None else checkpoint
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.cuda()
    sam_predictor = SamPredictor(sam)
    return grounding_dino, sam_predictor


@torch.no_grad()
def compute_masks_grounded(
    grounded_masks: list[torch.Tensor],
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    labels: torch.Tensor,
    mask_threshold: float,
    ioa_threshold: float,
):
    """Compute object masks based on the rendered masks and grounded masks."""
    labels_: torch.Tensor = labels.unique()
    unique_labels = labels_[labels_ != -1]

    results: list[torch.Tensor] = []
    for obj_label in unique_labels:
        mask_color = torch.tensor(
            [[1.0, 1.0, 1.0]], device=gaussians._xyz.device
        ).repeat(labels.shape[0], 1)
        label_mask = labels == obj_label
        mask_color[~label_mask] = 0
        renders, _ = render_frames(
            cameras,
            gaussians,
            model_params,
            gaussians._xyz.device,
            override_color=mask_color,
        )
        renders = renders.mean(dim=1) > mask_threshold  # (B, 3, H, W)
        results.append(renders)

    obj_masks = torch.stack(results, dim=1)

    # objects are selected only based on first frame
    # object masks need to be contiguous
    obj_ids, _ = select_objects_by_ioa(
        obj_masks[0], grounded_masks, ioa_threshold
    )  # list of length P with tensors of shape (ids)

    results: list[torch.Tensor] = []
    for ids in obj_ids:
        if obj_masks.ndim == 3:
            mask = obj_masks.unsqueeze(1) == ids.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1
            )
            mask = mask.any(dim=1)  # (B, H, W)
        else:
            mask = obj_masks.permute(1, 0, 2, 3).contiguous()[ids]
            mask = mask.any(dim=0)  # (B, H, W)
        results.append(mask)

    return torch.stack(results, dim=1)  # (B, P, H, W)


def get_grounded_masks(
    image: npt.NDArray[np.uint8],
    query_prompts: list[str],
    device: torch.device,
    grounded_models: tuple[GroundingDINO, SamPredictor] | None = None,
):
    if grounded_models is not None:
        grounding_dino, sam_predictor = grounded_models
    else:
        print("\nLoad GroundingDINO and SAM")
        grounding_dino, sam_predictor = setup_grounded_sam()

    grounded_masks: list[torch.Tensor] = []
    for prompt in tqdm(query_prompts, total=len(query_prompts), desc="Grounding SAM"):
        grounded_mask = grounded_sam_output(
            grounding_dino,
            sam_predictor,
            prompt,
            image,
            box_threshold=0.3,
            text_threshold=0.45,
            device=device,
        )
        grounded_masks.append(grounded_mask.cpu())
    return grounded_masks


def eval_lerf_mask_metrics(
    test_image_dirs: list[Path],
    query_prompts: list[str],
    pred_masks: npt.NDArray,
):
    iou_scores: dict[str, list[float]] = {}
    biou_scores: dict[str, list[float]] = {}
    class_counts: dict[str, int] = {}

    for i, image_dir in enumerate(test_image_dirs):
        for mask_file in image_dir.iterdir():
            prompt = mask_file.stem
            prompt_idx = query_prompts.index(prompt)
            gt_mask = np.asarray(iio.imread(mask_file), dtype=bool)
            pred_mask: npt.NDArray = pred_masks[i, prompt_idx]  # (H, W)
            assert gt_mask.shape == pred_mask.shape, (
                f"Shape mismatch: {gt_mask.shape} != {pred_mask.shape}"
            )

            iou = calculate_iou(gt_mask, pred_mask)
            biou = calculate_biou(gt_mask, pred_mask)

            if prompt not in iou_scores:
                iou_scores[prompt] = []
                biou_scores[prompt] = []
                class_counts[prompt] = 0
            iou_scores[prompt].append(iou)
            biou_scores[prompt].append(biou)
            class_counts[prompt] += 1

    miou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
    mbiou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}
    miou = np.mean(list(miou_per_class.values())) if miou_per_class else 0.0
    mbiou = np.mean(list(mbiou_per_class.values())) if mbiou_per_class else 0.0

    return {
        "miou_per_class": miou_per_class,
        "mbiou_per_class": mbiou_per_class,
        "miou": miou,
        "mbiou": mbiou,
    }


def eval_grounded(
    model_paths: list[Path],
    ioa_threshold: float = 0.7,
    mask_threshold: float = 0.2,
):
    scene_model_configs = get_scene_model_configs(model_paths)
    with warnings.catch_warnings(action="ignore"):
        grounded_models = setup_grounded_sam()

    for scene_name, configs in scene_model_configs.items():
        scene_config = LERF_MASK_SCENE_CONFIGS.get(scene_name, None)
        assert scene_config is not None, f"Scene {scene_name} not found in config"
        query_prompts = scene_config.prompts
        for model_path, config in configs:
            model_params = config.model
            model_params.eval = True  # Load test images
            # Do not load mask annotations
            model_params.mask_subdir = None

            setup_params = setup_from_config(config)

            mask_dir = Path(model_params.source_path) / "test_mask"
            assert mask_dir.exists(), (
                f"Directory {mask_dir} does not exist. It is not a lerf-mask scene."
            )
            assert mask_dir.is_dir(), f"Path {mask_dir} is not a directory"

            # Align test images with test cameras
            test_cameras = setup_params.scene.get_test_cameras()
            test_image_dirs = sorted(list(mask_dir.iterdir()))
            test_image_cameras = [
                next(c for c in test_cameras if c.name == f"test_{int(d.stem)}")
                for d in test_image_dirs
            ]

            labels = torch.from_numpy(np.load(model_path / "clustering" / "labels.npy"))
            images = test_image_cameras[0].original_image.unsqueeze(0)
            image: npt.NDArray = (images[0].permute(1, 2, 0) * 255).numpy()
            image = image.astype(np.uint8)

            with warnings.catch_warnings(action="ignore"):
                grounded_masks = get_grounded_masks(
                    image,
                    query_prompts,
                    setup_params.device,
                    grounded_models,
                )

            pred_masks: npt.NDArray = compute_masks_grounded(
                grounded_masks,
                test_image_cameras,
                setup_params.gaussians,
                model_params,
                labels,
                mask_threshold,
                ioa_threshold,
            ).numpy()
            assert pred_masks.ndim == 4, (
                f"Expected 4D array (B, P, H, W), got {pred_masks.ndim}D"
            )

            metrics = eval_lerf_mask_metrics(test_image_dirs, query_prompts, pred_masks)
            yield model_path, config, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_dir", type=Path, help="Path to the model or eval directory"
    )
    parser.add_argument(
        "--ioa-threshold",
        type=float,
        default=0.7,
        help="Threshold for IoA between object masks and grounded masks",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.3,
        help="Threshold for binarizing rendered masks",
    )
    parser.add_argument(
        "--per-scene",
        action="store_true",
        help="Evaluate also per scene if run directory",
    )

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Provided directory {exp_dir} does not exist")
        exit()

    is_eval_dir = (exp_dir / "scenes").exists()

    if is_eval_dir:
        model_dirs = [
            get_latest_model(model_dir)
            for model_dir in sorted((exp_dir / "scenes").iterdir())
        ]
    else:
        if not (exp_dir / "config.yaml").exists():
            exp_dir = get_latest_model(exp_dir)
        model_dirs = [exp_dir]

    if len(model_dirs) == 0:
        print("\nNo models found")
        exit()

    print(f"\nFound {len(model_dirs)} models")

    eval_subdir = "eval_results/grounded"
    output_name = generate_eval_output_name()

    eval_cfg = OmegaConf.create(
        {
            "ioa_threshold": args.ioa_threshold,
            "mask_threshold": args.mask_threshold,
        }
    )

    results: list[tuple[Path, Config, dict[str, typing.Any]]] = []
    for result in eval_grounded(
        model_dirs,
        args.ioa_threshold,
        args.mask_threshold,
    ):
        if is_eval_dir:
            results.append(result)
            if not args.per_scene:
                continue
        # Store results in all model directories
        output_path = result[0] / eval_subdir / output_name
        output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(eval_cfg, output_path / "eval_config.yaml")
        save_lerf_mask_results([result], output_path)

    if is_eval_dir and len(results) > 0:
        run_output_path = exp_dir / eval_subdir / output_name
        run_output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(eval_cfg, run_output_path / "eval_config.yaml")
        save_lerf_mask_results(results, run_output_path)
