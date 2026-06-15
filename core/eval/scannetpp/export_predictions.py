import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from cuml.neighbors import NearestNeighbors
from omegaconf import OmegaConf

from core.cluster.hdbscan import dbscan_denoising
from core.config import config_from_yaml
from core.eval.scannetpp.label_utils import load_label_infos
from core.gaussian_model import GaussianModel, sub_gaussians
from core.language import LanguageModel
from core.utils.rle_utils import rle_encode
from core.utils.setup_utils import get_latest_model, setup

SCANNETPP_ROOT_PATH = (
    Path(os.environ["SCANNETPP_ROOT_PATH"])
    if "SCANNETPP_ROOT_PATH" in os.environ
    else None
)
SCANNETPP_DATA_PATH = (
    SCANNETPP_ROOT_PATH / "data" if SCANNETPP_ROOT_PATH is not None else None
)
SCANNETPP_PTH_PATH = (
    Path(os.environ["SCANNETPP_PTH_PATH"])
    if "SCANNETPP_PTH_PATH" in os.environ
    else (
        (Path(os.environ["SCANNETPP_PATH"]) / "pths" / "val")
        if "SCANNETPP_PATH" in os.environ
        else None
    )
)
SCANNETPP_SEGMENTS_PATH = (
    Path(os.environ["SCANNETPP_SEGMENTS_PATH"])
    if "SCANNETPP_SEGMENTS_PATH" in os.environ
    else (
        (Path(os.environ["SCANNETPP_PATH"]) / "segments")
        if "SCANNETPP_PATH" in os.environ
        else None
    )
)


@dataclass
class DBSCANParams:
    min_samples: int = 5
    eps: float = 0.5
    min_cluster_size: int = 0


@dataclass
class Segments:
    unique_seg_indices: npt.NDArray[np.int32]
    points_to_segments_indices: npt.NDArray[np.int32]


@torch.no_grad()
def assign_points_to_gaussians(
    gaussians: GaussianModel, points: npt.NDArray[np.float32], k: int = 1
):
    """
    Assigns the closest Gaussian index for each evaluation point based on the given distance metric.

    Args:
        gaussians: The Gaussian model.
        points: The evaluation points.
    """
    xyz = gaussians.get_xyz.detach()
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(xyz)
    _, indices = nn.kneighbors(points, return_distance=True)
    return torch.as_tensor(indices, dtype=torch.int32, device=xyz.device)


def filter_valid_gaussians(
    gaussians: GaussianModel,
    cluster_labels: npt.NDArray[np.int32],
    valid_clusters_mask: npt.NDArray[np.bool_],
) -> tuple[GaussianModel, npt.NDArray[np.bool_]]:
    # valid_clusters is numpy array that contains boolean value per unique cluster id
    # Remove all gaussians belonging to invalid clusters
    valid_gaussians_mask = cluster_labels != -1
    unique_labels, _ = np.unique(cluster_labels, return_counts=True)
    unique_labels = unique_labels[unique_labels != -1]
    assert unique_labels.shape[0] == valid_clusters_mask.shape[0]
    for label, valid in zip(unique_labels, valid_clusters_mask):
        if not valid:
            valid_gaussians_mask[cluster_labels == label] = False

    if unique_labels[~valid_clusters_mask].shape[0] > 0:
        print(
            "Removed invalid clusters:",
            unique_labels[~valid_clusters_mask].shape,
            unique_labels[~valid_clusters_mask],
        )
    return sub_gaussians(gaussians, valid_gaussians_mask), valid_gaussians_mask


def assign_clusters_to_points(
    gaussians: GaussianModel,
    points: npt.NDArray[np.float32],
    k: int,
    cluster_labels: npt.NDArray[np.int32],
    valid_clusters_mask: npt.NDArray[np.bool_],
):
    valid_gaussians, valid_gaussians_mask = filter_valid_gaussians(
        gaussians,
        cluster_labels,
        valid_clusters_mask,
    )

    indices = assign_points_to_gaussians(valid_gaussians, points, k)

    cluster_labels_torch = torch.from_numpy(cluster_labels).to(device=indices.device)
    cluster_labels_torch = cluster_labels_torch[valid_gaussians_mask][indices]
    mode_result = torch.mode(cluster_labels_torch, dim=-1)
    return mode_result.values


def postprocess_with_segments(
    pred_labels: torch.Tensor,
    points_to_segments_indices: npt.NDArray[np.int32],
):
    pred_labels_ = pred_labels.clone()
    for seg_id in np.unique(points_to_segments_indices):
        seg_mask = points_to_segments_indices == seg_id
        seg_pred_label = torch.mode(pred_labels_[seg_mask], dim=-1).values
        pred_labels_[seg_mask] = seg_pred_label
    return pred_labels_


def predict_instances(
    cluster_lang_embeds: torch.Tensor,
    pred_clusters: npt.NDArray[np.int32],
    inst_sem_text_embed: torch.Tensor,
    inst_sem_valid_class_ids: list[int],
    lang_model: LanguageModel,
    denoised_pred_clusters: npt.NDArray[np.int32] | None = None,
) -> tuple[list[dict], list[dict]]:
    unique_pred_clusters: npt.NDArray[np.int32] = np.unique(pred_clusters)

    # Instance prediction
    sim = cluster_lang_embeds @ inst_sem_text_embed.T
    sim = lang_model.rescale(sim)
    probs = torch.softmax(sim, dim=-1)
    pred_sem_labels = sim.argmax(dim=-1).cpu().numpy()
    conf = probs.max(dim=-1).values

    # Map instances (cluster_id) to semantic label (text id)
    pred_sem_dict: dict[int, int] = {
        cluster_id: pred_sem_labels[i]
        for i, cluster_id in enumerate(unique_pred_clusters)
    }

    inst_agnostic_pred_out = []
    inst_pred_out = []

    if denoised_pred_clusters is not None:
        unique_denoised_pred_clusters = np.unique(denoised_pred_clusters)
        for pred_cluster in unique_denoised_pred_clusters:
            mask: npt.NDArray[np.bool_] = denoised_pred_clusters == pred_cluster
            rle = rle_encode(mask)
            inst_agnostic_pred_out.append(
                {
                    "label": 0,
                    "confidence": 1.0,
                    "mask": rle,
                }
            )

    for i, pred_cluster in enumerate(unique_pred_clusters):
        mask: npt.NDArray[np.bool_] = pred_clusters == pred_cluster
        rle = rle_encode(mask)
        if denoised_pred_clusters is None:
            inst_agnostic_pred_out.append(
                {
                    "label": 0,
                    "confidence": 1.0,
                    "mask": rle,
                }
            )
        inst_pred_out.append(
            {
                "label": inst_sem_valid_class_ids[pred_sem_dict[pred_cluster]],
                "confidence": conf[i],
                "mask": rle,
            }
        )
    return inst_pred_out, inst_agnostic_pred_out


def predict_semantics(
    cluster_lang_embeds: torch.Tensor,
    pred_clusters: npt.NDArray[np.int32],
    sem_text_embed: torch.Tensor,
    lang_model: LanguageModel,
    topk: int = 3,
) -> dict:
    unique_pred_clusters: npt.NDArray[np.int32] = np.unique(pred_clusters)
    sim = cluster_lang_embeds @ sem_text_embed.T
    sim = lang_model.rescale(sim)
    probs = lang_model.activation(sim)
    topk_result = probs.topk(k=topk, dim=-1)
    pred_sem_labels = topk_result.indices.cpu().numpy()
    conf = topk_result.values.sigmoid().cpu().numpy()

    pred_sem = np.ones((pred_clusters.shape[0], topk), dtype=np.int32) * -1
    pred_conf = np.zeros((pred_clusters.shape[0], topk))
    for i, cluster_id in enumerate(unique_pred_clusters):
        mask = pred_clusters == cluster_id
        pred_sem[mask] = pred_sem_labels[i]
        pred_conf[mask] = conf[i]

    assert pred_sem.min() != -1, "All instances must have a semantic label."

    sem_pred_out = {
        "labels": pred_sem,
        "conf": pred_conf,
    }
    return sem_pred_out


def write_instance_predictions(
    predictions: list[dict], output_path: Path, scene_name: str
):
    mask_dir = output_path / "predicted_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{scene_name}.txt", "w") as f1:
        for i, x in enumerate(predictions):
            mask_file = mask_dir / f"{scene_name}_{i:03d}.json"
            with open(mask_file, "w") as f2:
                json.dump(x["mask"], f2)
            f1.write(
                f"{mask_file.relative_to(output_path)} {x['label']} {x['confidence']}\n"
            )


def load_segments(scene_name: str) -> Segments:
    assert SCANNETPP_SEGMENTS_PATH is not None, "SCANNETPP_SEGMENTS_PATH is not set."
    segments_path = SCANNETPP_SEGMENTS_PATH / f"{scene_name}.0.010000.segs.json"
    assert segments_path.exists(), f"Segments file not found: {segments_path}"
    with open(segments_path, "r") as f:
        segments = json.load(f)
        seg_indices = np.array(segments["segIndices"])
        _, unique_seg_indices, points_to_segments_indices = np.unique(
            seg_indices, return_index=True, return_inverse=True
        )
    return Segments(
        unique_seg_indices=unique_seg_indices,
        points_to_segments_indices=points_to_segments_indices,
    )


def export_scene_prediction(
    model_path: Path,
    output_path: Path,
    scene_id: str,
    lang_model: LanguageModel,
    sem_text_embed: torch.Tensor,
    inst_text_embed: torch.Tensor,
    inst_sem_valid_class_ids: list[int],
    knn_k: int,
    sem_topk: int = 3,
    use_segments: bool = False,
    dbscan_params: DBSCANParams | None = None,
):
    assert SCANNETPP_DATA_PATH is not None, "SCANNETPP_DATA_PATH is not set."
    assert SCANNETPP_PTH_PATH is not None, "SCANNETPP_PTH_PATH is not set."

    scene_dir = SCANNETPP_DATA_PATH / scene_id
    pth_file = SCANNETPP_PTH_PATH / f"{scene_id}.pth"

    assert scene_dir.exists(), f"Scene directory not found: {scene_dir}"
    assert pth_file.exists(), f"Scene .pth file not found: {pth_file}"

    points = torch.load(pth_file, weights_only=False)["vtx_coords"]

    segments = load_segments(scene_dir.name) if use_segments else None

    setup_params = setup(model_path, load_masks=False)

    cluster_labels = np.load(
        Path(setup_params.model_params.model_path) / "clustering" / "labels.npy"
    )

    embeds_path = (
        Path(setup_params.model_params.model_path)
        / f"{lang_model.model_type}_embeddings.pth"
    )
    assert embeds_path.exists(), f"Embeddings file {embeds_path} does not exist."
    print()
    print("Loading embeddings from:", embeds_path)
    print()
    embeddings_info = torch.load(embeds_path, weights_only=False)

    pred_clusters = assign_clusters_to_points(
        setup_params.gaussians, points, knn_k, cluster_labels, embeddings_info["valid"]
    )
    if segments is not None:
        print("Using segments for post-processing.")
        pred_clusters = postprocess_with_segments(
            pred_clusters, segments.points_to_segments_indices
        )

    pred_clusters = pred_clusters.cpu().numpy()

    denoised_pred_clusters = None
    if dbscan_params is not None:
        print(
            f"Denoising predictions with DBSCAN: min_samples={dbscan_params.min_samples}, eps={dbscan_params.eps}, min_cluster_size={dbscan_params.min_cluster_size}"
        )
        denoised_pred_clusters = dbscan_denoising(
            points,
            pred_clusters,
            dbscan_params.min_samples,
            dbscan_params.eps,
            dbscan_params.min_cluster_size,
        )

    # Compute instance language embeddings only for predicted instances (unique_pred_labels)
    unique_pred_clusters = np.unique(pred_clusters)
    inst_ids = unique_pred_clusters[unique_pred_clusters != -1]
    cluster_lang_embeds = torch.from_numpy(
        embeddings_info["embeddings"][inst_ids]
    ).cuda()

    # Semantic predictions
    sem_preds = predict_semantics(
        cluster_lang_embeds,
        pred_clusters,
        sem_text_embed,
        lang_model,
        sem_topk,
    )

    sem_output_path = output_path / "semantic"
    sem_output_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        sem_output_path / f"{scene_dir.name}.txt",
        sem_preds["labels"],
        fmt="%d",
        delimiter=",",
    )
    print()
    print("Semantic predictions saved to:", sem_output_path / f"{scene_dir.name}.txt")

    # Instance and agnostic instance predictions
    inst_preds, inst_agnostic_preds = predict_instances(
        cluster_lang_embeds,
        pred_clusters,
        inst_text_embed,
        inst_sem_valid_class_ids,
        lang_model,
        denoised_pred_clusters,
    )

    inst_output_path = output_path / "instance"
    write_instance_predictions(inst_preds, inst_output_path, scene_dir.name)
    print()
    print("Instance predictions saved to:", inst_output_path / f"{scene_dir.name}.txt")

    inst_output_path = output_path / "instance_agnostic"
    write_instance_predictions(inst_agnostic_preds, inst_output_path, scene_dir.name)
    print()
    print(
        "Agnostic instance predictions saved to:",
        inst_output_path / f"{scene_dir.name}.txt",
    )


def export_scenes_predictions(
    model_paths: list[Path],
    output_path: Path,
    lang_model_type: str,
    knn_k: int,
    sem_topk: int = 3,
    use_segments: bool = False,
    dbscan_params: DBSCANParams | None = None,
):
    assert SCANNETPP_ROOT_PATH is not None, "SCANNETPP_ROOT_PATH is not set."

    export_cfg = OmegaConf.create(
        {
            "lang_model": lang_model_type,
            "knn_k": knn_k,
            "sem_topk": sem_topk,
            "segments": use_segments,
            "dbscan": {
                "min_samples": dbscan_params.min_samples,
                "eps": dbscan_params.eps,
                "min_cluster_size": dbscan_params.min_cluster_size,
            }
            if dbscan_params is not None
            else None,
        }
    )

    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(export_cfg, output_path / "export_config.yaml")

    scene_ids = (
        (SCANNETPP_ROOT_PATH / "splits" / "nvs_sem_val.txt")
        .read_text()
        .strip()
        .splitlines()
    )

    sem_label_info, inst_label_info = load_label_infos(SCANNETPP_ROOT_PATH)

    print(f"Loading {lang_model_type} model")
    lang_model = LanguageModel(lang_model_type)

    sem_text_embed = lang_model.embed_text(
        [lang_model.prompt_template.format(x) for x in sem_label_info.class_labels],
        normalize=True,
    )
    inst_text_embed = lang_model.embed_text(
        [lang_model.prompt_template.format(x) for x in inst_label_info.class_labels],
        normalize=True,
    )

    inst_sem_valid_class_ids = inst_label_info.valid_class_ids

    for model_path in model_paths:
        if not (model_path / "config.yaml").exists():
            model_path = get_latest_model(model_path)

        config = config_from_yaml(model_path / "config.yaml")
        # Try to get scene_id from source_path in config
        scene_id = Path(config.model.source_path).name
        if scene_id not in scene_ids:
            # Fallback: get scene_id from model_path
            scene_id = model_path.parts[-2]

        print(f"\nProcessing scene '{scene_id}' with model:", model_path)

        assert scene_id in scene_ids, f"Scene {scene_id} not in validation set."

        export_scene_prediction(
            model_path,
            output_path,
            scene_id,
            lang_model,
            sem_text_embed,
            inst_text_embed,
            inst_sem_valid_class_ids,
            knn_k,
            sem_topk,
            use_segments,
            dbscan_params,
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
    parser.add_argument("--knn-k", type=int, default=1)
    parser.add_argument(
        "--sem-topk",
        type=int,
        default=3,
        help="Output top-k semantic labels. ScanNet++ evaluation supports top-1 and top-3.",
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Use segments instead of points (.segs.json).",
    )
    parser.add_argument("--dbscan", action="store_true")
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--dbscan-eps", type=float, default=0.5)
    parser.add_argument("--dbscan-min-cluster-size", type=int, default=0)

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

    output_path = (
        exp_dir
        / "eval_predictions"
        / f"{'segments' if args.segments else 'without-postprocessing'}"
    )

    dbscan_params = (
        DBSCANParams(
            args.dbscan_min_samples,
            args.dbscan_eps,
            args.dbscan_min_cluster_size,
        )
        if args.dbscan
        else None
    )

    export_scenes_predictions(
        model_dirs,
        output_path,
        args.lang_model,
        args.knn_k,
        args.sem_topk,
        args.segments,
        dbscan_params,
    )
