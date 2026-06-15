# Evaluate LERF-OVS dataset (object localization)
# Code copied from OpenGaussian (https://github.com/yanmin-wu/OpenGaussian/blob/main/scripts/compute_lerf_iou.py)


import json
import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import tabulate
import torch
from omegaconf import OmegaConf

from core.config import Config
from core.eval.metrics import calculate_iou, calculate_loc
from core.eval.utils import (
    generate_eval_output_name,
    get_scene_model_configs,
    render_alpha,
    render_frames,
)
from core.gaussian_model import GaussianModel, sub_gaussians
from core.language import LanguageModel
from core.params import ModelParams
from core.scene.camera import Camera
from core.utils.setup_utils import get_latest_model, setup_from_config

LERF_OVS_LABEL_PATH = (
    Path(os.environ["LERF_OVS_LABEL_PATH"])
    if "LERF_OVS_LABEL_PATH" in os.environ
    else None
)


@dataclass
class ObjectAnnotation:
    label: str
    bboxes: npt.NDArray[np.float32]
    masks: npt.NDArray[np.bool_]


@dataclass
class FrameAnnotation:
    frame: Path
    annotations: dict[str, ObjectAnnotation]  # label -> ObjectAnnotations


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)  # type: ignore
    return mask.astype(bool)


def get_annotations(label_scene_path: Path):
    annotations: list[FrameAnnotation] = []
    frames = sorted(label_scene_path.glob("*.jpg"))
    for frame in frames:
        anno: dict[str, dict[str, list[npt.NDArray]]] = {}
        with open(frame.with_suffix(".json"), "r") as f:
            json_data = json.load(f)
        h = json_data["info"]["height"]
        w = json_data["info"]["width"]
        objects = json_data["objects"]
        for obj in objects:
            label = obj["category"]
            bbox = np.asarray(obj["bbox"])
            mask = polygon_to_mask((h, w), obj["segmentation"])
            if label not in anno:
                anno[label] = {"bboxes": [], "masks": []}
            anno[label]["bboxes"].append(bbox)
            anno[label]["masks"].append(mask)
        annotation = {
            k: ObjectAnnotation(k, np.stack(v["bboxes"]), np.stack(v["masks"]))
            for k, v in anno.items()
        }
        annotations.append(FrameAnnotation(frame, annotation))

    scene_labels = sorted(set([k for f in annotations for k in f.annotations.keys()]))

    return annotations, scene_labels


def lerf_ovs_metrics_table_str(metrics: dict[str, Any]):
    table = [("Category", "IoU", "Acc", "Loc")]
    table.extend(
        [
            (
                cat_id,
                f"{metrics['classes']['average_iou'][cat_id] * 100:.2f}",
                f"{metrics['classes']['acc_025'][cat_id] * 100:.2f}",
                f"{metrics['classes']['average_loc'][cat_id] * 100:.2f}",
            )
            for cat_id in metrics["classes"]["average_iou"]
        ]
    )
    table.append(
        (
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
        )
    )
    table.append(
        (
            "Mean",
            f"{metrics['all']['average_iou'] * 100:.2f}",
            f"{metrics['all']['acc_025'] * 100:.2f}",
            f"{metrics['all']['average_loc'] * 100:.2f}",
        )
    )
    table_str = tabulate.tabulate(table, headers="firstrow", floatfmt=".2f")
    return table_str


def save_lerf_ovs_results(
    results: list[tuple[Path, Config, dict]],
    output_path: Path,
    save_txt: bool = True,
    save_json: bool = True,
    print_summary: bool = True,
):
    output_json = {"scenes": {}, "all": {}}
    log_lines: list[str] = []
    all_metrics = {"miou": 0.0, "macc": 0.0, "mloc": 0.0}
    table = [("Scene", "mIoU", "mAcc", "mLoc")]

    for model_path, _, metrics in results:
        log_lines.extend(["\n"])
        # model_path = Path(config.model.model_path)
        model_name = model_path.relative_to(model_path.parent.parent)
        log_lines.extend([f"{model_name}\n"])
        table_str = lerf_ovs_metrics_table_str(metrics)
        log_lines.extend([table_str, "\n"])
        scene_name = model_path.parent.parent.name
        all_metrics["miou"] += metrics["all"]["average_iou"]
        all_metrics["macc"] += metrics["all"]["acc_025"]
        all_metrics["mloc"] += metrics["all"]["average_loc"]
        table.append(
            (
                str(scene_name),
                f"{metrics['all']['average_iou'] * 100:.2f}",
                f"{metrics['all']['acc_025'] * 100:.2f}",
                f"{metrics['all']['average_loc'] * 100:.2f}",
            )
        )
        output_json["scenes"][scene_name] = metrics

    if len(results) > 0:
        all_metrics["miou"] /= len(results)
        all_metrics["macc"] /= len(results)
        all_metrics["mloc"] /= len(results)

    table.append(
        (
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
            tabulate.SEPARATING_LINE,
        )
    )
    table.append(
        (
            "Mean",
            f"{all_metrics['miou'] * 100:.2f}",
            f"{all_metrics['macc'] * 100:.2f}",
            f"{all_metrics['mloc'] * 100:.2f}",
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


def evaluate_lerf_ovs_metrics(
    gt_annotations: list[FrameAnnotation],
    predictions: dict[str, dict[str, npt.NDArray[np.bool_]]],
):
    assert len(gt_annotations) == len(predictions), (
        "Predictions and GT must have same length"
    )
    ious: list[float] = []
    locs: list[bool] = []
    class_iou: dict[str, list[float]] = {}
    class_loc: dict[str, list[bool]] = {}
    for frame_anno in gt_annotations:
        frame_preds = predictions[frame_anno.frame.stem]
        for label, anno in frame_anno.annotations.items():
            mask_gt: npt.NDArray[np.bool_] = anno.masks.sum(
                axis=0
            )  # merge masks of same category
            bboxes_gt = anno.bboxes
            mask_pred = frame_preds[label]
            iou = calculate_iou(mask_gt, mask_pred)
            loc = calculate_loc(bboxes_gt, mask_pred)
            if label not in class_iou:
                class_iou[label] = []
            class_iou[label].append(iou)
            if label not in class_loc:
                class_loc[label] = []
            class_loc[label].append(loc)
            ious.append(iou)
            locs.append(loc)

    # Acc.
    total_count = len(ious)
    count_iou_025 = (np.array(ious) > 0.25).sum()
    count_iou_05 = (np.array(ious) > 0.5).sum()

    class_iou_avg = {label: np.mean(iou) for label, iou in class_iou.items()}
    class_acc_025 = {
        label: (np.array(iou) > 0.25).sum() / len(iou)
        for label, iou in class_iou.items()
    }
    class_acc_05 = {
        label: (np.array(iou) > 0.5).sum() / len(iou)
        for label, iou in class_iou.items()
    }

    class_loc_avg = {label: np.mean(loc) for label, loc in class_loc.items()}

    # mIoU
    average_iou = np.mean(ious)
    average_loc = np.mean(locs)
    return {
        "classes": {
            "average_iou": class_iou_avg,
            "acc_025": class_acc_025,
            "acc_05": class_acc_05,
            "average_loc": class_loc_avg,
        },
        "all": {
            "average_iou": average_iou,
            "average_loc": average_loc,
            "acc_025": count_iou_025 / total_count,
            "acc_05": count_iou_05 / total_count,
        },
    }


@torch.no_grad()
def get_instances_from_language(
    text_embeddings: torch.Tensor,
    lang_embeddings: torch.Tensor,
    lang_model: LanguageModel,
    labels: torch.Tensor,
    pred_threshold: float,
    pred_type: str = "max",
):
    assert labels.unique().shape[0] - 1 == lang_embeddings.shape[0], (
        labels.unique().shape,
        lang_embeddings.shape,
    )
    unique_labels: torch.Tensor = labels.unique()
    unique_labels = unique_labels[unique_labels != -1]
    assert unique_labels.shape[0] == lang_embeddings.shape[0]
    # labels contains ignore label -1
    sim = lang_embeddings @ text_embeddings.T
    sim = sim.float()

    if pred_type == "max":
        sim = lang_model.rescale(sim)
        pred = sim
        pred = pred - pred.min(dim=0, keepdim=True).values
        mask = pred >= pred.max(dim=0, keepdim=True).values * pred_threshold
    elif pred_type == "output":
        sim = lang_model.rescale(sim)
        pred = lang_model.activation(sim)
        mask = pred >= pred_threshold
    elif pred_type == "cosine":
        pred = sim
        mask = sim >= pred_threshold
    else:
        raise ValueError(f"Invalid pred_type: {pred_type}")

    mask = mask.T  # (P, I)
    pred_ids = [x.nonzero(as_tuple=True)[0] for x in mask]
    obj_ids = [unique_labels[ids] for ids in pred_ids]
    return obj_ids, pred


@torch.no_grad()
def compute_masks_inst_3d(
    text_embeddings: torch.Tensor,
    lang_embeddings: torch.Tensor,
    lang_model: LanguageModel,
    cameras: list[Camera],
    gaussians: GaussianModel,
    model_params: ModelParams,
    labels: torch.Tensor,
    pred_threshold: float,
    mask_threshold: float,
    device: torch.device,
    pred_type: str = "max",
    render_mode: str | None = None,
):
    """
    Compute instance masks for the LERF-Mask and LERF-OVS dataset using instance language embeddings.
    Args:
        text_embeddings: Text embeddings for instance language queries
        lang_embeddings: Instance language embeddings
        lang_model: Language model
        cameras: Cameras for rendering
        gaussians: Gaussian model
        model_params: Model parameters
        labels: Instance labels
        pred_threshold: Prediction threshold
        mask_threshold: Mask threshold
        device: Device
        pred_type: Prediction type
        render_mode: Render instances isolated without occlusion, only alpha mask or normal
    """
    assert render_mode in {None, "isolated", "alpha"}, (
        f"Invalid render_mode: {render_mode}"
    )
    obj_ids, _ = get_instances_from_language(
        text_embeddings, lang_embeddings, lang_model, labels, pred_threshold, pred_type
    )

    results: list[torch.Tensor] = []
    for ids in obj_ids:
        mask_color = torch.tensor([[1.0, 1.0, 1.0]], device=device).repeat(
            labels.shape[0], 1
        )
        mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
        for id in ids:
            mask |= labels == id
        mask_color[~mask] = 0
        if render_mode in {"isolated", "alpha"}:
            G = sub_gaussians(gaussians, mask)  # type: ignore
            mask_color = mask_color[mask]
        else:
            G = gaussians
        if render_mode == "alpha":
            renders = render_alpha(cameras, G, model_params, device)
            renders = renders > mask_threshold
        else:
            renders, _ = render_frames(
                cameras,
                G,
                model_params,
                device,
                override_color=mask_color,
            )
            renders = renders.mean(dim=1) > mask_threshold
        results.append(renders)

    return torch.stack(results, dim=1)  # (B, P, H, W)


@torch.no_grad()
def eval_lerf_ovs_inst_3d(
    model_paths: list[Path],
    lang_model_type: str,
    pred_threshold: float,
    mask_threshold: float = 0.2,
    pred_type: str = "max",
    render_mode: str | None = None,
):
    assert LERF_OVS_LABEL_PATH is not None, "LERF_OVS_LABEL_PATH is not set."

    scene_model_configs = get_scene_model_configs(model_paths)
    lang_model = LanguageModel(lang_model_type)

    for scene_name, configs in scene_model_configs.items():
        scene_label_path = LERF_OVS_LABEL_PATH / scene_name
        gt_annotations, query_prompts = get_annotations(scene_label_path)
        test_frame_names = [x.frame.stem for x in gt_annotations]

        text_embeddings = lang_model.embed_text(
            [lang_model.prompt_template.format(x) for x in query_prompts],
            normalize=True,
        )

        for model_path, config in configs:
            model_params = config.model
            model_params.eval = True  # Load test images
            # Do not load mask annotations
            model_params.mask_subdir = None

            setup_params = setup_from_config(config, model_path)

            # Align test images with test cameras
            test_cameras = setup_params.scene.get_test_cameras()
            test_image_cameras = [
                next(c for c in test_cameras if c.name == test_name)
                for test_name in test_frame_names
            ]

            lang_embeddings = []
            valid_labels = torch.from_numpy(
                np.load(model_path / "clustering" / "labels.npy")
            )
            embeds_path = model_path / f"{lang_model_type}_embeddings.pth"
            embeds_data: dict = torch.load(embeds_path, weights_only=False)
            embeds_data.pop("config", None)

            valid_clusters = embeds_data["valid"]
            lang_embeddings = torch.from_numpy(
                embeds_data["embeddings"][valid_clusters]
            ).cuda()
            lang_embeddings = lang_embeddings / lang_embeddings.norm(
                dim=-1, keepdim=True
            )
            unique_labels = torch.unique(valid_labels)
            unique_labels = unique_labels[unique_labels != -1]
            assert unique_labels.shape[0] == valid_clusters.shape[0]
            for label, valid in zip(unique_labels, valid_clusters):
                if not valid:
                    valid_labels[valid_labels == label] = -1

            pred_masks: npt.NDArray = compute_masks_inst_3d(
                text_embeddings,
                lang_embeddings,
                lang_model,
                test_image_cameras,
                setup_params.gaussians,
                model_params,
                valid_labels.cuda(),
                pred_threshold,
                mask_threshold,
                setup_params.device,
                pred_type,
                render_mode,
            ).numpy()
            assert pred_masks.ndim == 4, (
                f"Expected 4D array (B, P, H, W), got {pred_masks.ndim}D"
            )

            predictions = {
                test_cam.name: {
                    prompt: obj_mask
                    for prompt, obj_mask in zip(query_prompts, frame_masks)
                }
                for test_cam, frame_masks in zip(test_image_cameras, pred_masks)
            }
            metrics = evaluate_lerf_ovs_metrics(gt_annotations, predictions)
            yield (
                model_path,
                config,
                predictions,
                metrics,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", type=Path, help="Path to the model or eval directory"
    )
    parser.add_argument(
        "--lang-model",
        type=str,
        default="masqclip",
        choices=["clip", "siglip", "masqclip"],
        help="Language model",
    )
    parser.add_argument(
        "--pred-type",
        type=str,
        default="max",
        choices=["max", "output", "cosine"],
        help="Prediction type",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.85,
        help="Threshold for instance language query",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.3,
        help="Threshold for binarizing rendered masks",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=["alpha", "isolated"],
        help="Render mode for masks. Default is with occlusion.",
    )
    parser.add_argument(
        "--per-scene",
        action="store_true",
        help="Evaluate also per scene if run directory",
    )

    args = parser.parse_args()

    exp_dir = Path(args.model_dir)
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

    eval_subdir = f"eval_results/inst3d{f'_{args.render_mode}' if args.render_mode is not None else ''}"
    output_name = generate_eval_output_name()

    eval_cfg = OmegaConf.create(
        {
            "lang_model": args.lang_model,
            "pred_type": args.pred_type,
            "pred_threshold": args.pred_threshold,
            "mask_threshold": args.mask_threshold,
            "render_mode": args.render_mode,
        }
    )

    results: list[tuple[Path, Config, dict[str, typing.Any]]] = []

    for result in eval_lerf_ovs_inst_3d(
        model_dirs,
        args.lang_model,
        args.pred_threshold,
        args.mask_threshold,
        args.pred_type,
        args.render_mode,
    ):
        if is_eval_dir:
            results.append((result[0], result[1], result[3]))
            if not args.per_scene:
                continue
        # Store results in all model directories
        output_path = result[0] / eval_subdir / output_name
        output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(eval_cfg, output_path / "eval_config.yaml")
        save_lerf_ovs_results([(result[0], result[1], result[3])], output_path)

    if is_eval_dir and len(results) > 0:
        run_output_path = exp_dir / eval_subdir / output_name
        run_output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(eval_cfg, run_output_path / "eval_config.yaml")
        save_lerf_ovs_results(results, run_output_path)
