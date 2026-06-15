import os
import sys

# CRITICAL: Environment Isolation
# TITAN X (Pascal) crashes cuML kernels. 
# We force the process to only see the RTX 3060 Ti (Device 0) 
# to prevent cuML from probing the incompatible card.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
import sklearn.cluster as sklearn_cluster
import hdbscan as cpu_hdbscan

try:
    from cuml.cluster import DBSCAN as GPU_DBSCAN
    from cuml.cluster import HDBSCAN as GPU_HDBSCAN
except (ImportError, Exception):
    print("Warning: GPU Clustering via cuML disabled (expected on some architectures).")
    GPU_DBSCAN, GPU_HDBSCAN = None, None
from diff_gaussian_rasterization import MAX_FEATURE_DIM
from omegaconf import OmegaConf
from tqdm import tqdm

from core.gaussian_model import GaussianModel
from core.params import ModelParams
from core.utils.setup_utils import get_latest_model, setup_model
from core.utils.sh_utils import SH2RGB


def get_cluster_features(
    gaussians: GaussianModel,
    model_params: ModelParams,
    with_position: float = 0.0,
    with_color: float = 0.0,
):
    assert model_params.mask_dim > 0, "invalid mask dims"
    assert gaussians.get_features is not None, "features not available"
    cluster_features = (
        gaussians.get_features[..., : model_params.mask_dim].detach().cpu().numpy()
    )
    if with_color > 0:
        base_color = SH2RGB(
            gaussians.get_spherical_harmonics.detach().cpu().numpy()[..., 0, :]
        )
        cluster_features = np.concatenate(
            [base_color * with_color, cluster_features], axis=1
        )
    if with_position > 0:
        cluster_features = np.concatenate(
            [
                gaussians.get_xyz.detach().cpu().numpy() * with_position,
                cluster_features,
            ],
            axis=1,
        )
    return cluster_features


def dbscan_denoising(
    xyz: npt.NDArray,
    labels: npt.NDArray,
    min_samples: int,
    eps: float,
    min_cluster_size: int,
):
    labels_ = labels.copy()
    unique_labels = np.unique(labels_)
    max_label = np.max(labels_) + 1

    for label in tqdm(unique_labels):
        if label == -1:
            continue
        mask = labels_ == label
        cluster = xyz[mask]
        
        # Try GPU DBSCAN first
        try:
            if GPU_DBSCAN is not None:
                new_labels = GPU_DBSCAN(min_samples=min_samples, eps=eps).fit_predict(cluster)
            else:
                raise ImportError("GPU DBSCAN not available")
        except Exception as e:
            if "cudaErrorNoKernelImageForDevice" in str(e) or isinstance(e, ImportError):
                print(f"CUDA Error/Missing (Pascal logic): Falling back to CPU DBSCAN for label {label}...")
                new_labels = sklearn_cluster.DBSCAN(min_samples=min_samples, eps=eps).fit_predict(cluster)
            else:
                raise e

        # label -> -1, 0, 1, 2, ...
        # Set all cluster points to -1 and add new labels for new clusters
        labels_[mask] = -1

        for new_label in np.unique(new_labels):
            if new_label == -1:
                continue
            if np.sum(new_labels == new_label) < min_cluster_size:
                continue
            new_mask = mask.copy()
            new_mask[mask] = new_labels == new_label
            labels_[new_mask] = new_label + max_label

        max_label = np.max(new_labels) + max_label + 1

    for i, label in enumerate(np.unique(labels_), start=-1 if -1 in labels else 0):
        if label == -1:
            continue
        mask = labels_ == label
        labels_[mask] = i

    return labels_


def print_cluster_info(labels: npt.NDArray):
    unique, count = np.unique(labels, return_counts=True)
    print("#cluster:", len(unique))
    sort_index = np.argsort(count)[::-1]
    unique = unique[sort_index]
    count = count[sort_index]
    noisy_samples = int(np.sum(labels == -1))
    print("noisy samples:", noisy_samples, f"({noisy_samples / len(labels):.2%})")
    print("largest 20 labels:", unique[:20])
    print("largest 20 counts:", count[:20])


def hdbscan_clustering(
    model_path: Path,
    output_dir: Path,
    with_position: float,
    with_color: float,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    use_dbscan_denoising: bool = False,
    dbscan_min_samples: int = 5,
    dbscan_eps: float = 0.5,
    dbscan_min_cluster_size: int = 0,
):
    setup_params = setup_model(model_path)
    cfg = setup_params.config
    gaussians = setup_params.gaussians

    print(
        f"Using {cfg.model.mask_dim}/{MAX_FEATURE_DIM} dimensions, mask={cfg.model.mask_dim}"
    )

    cluster_cfg = OmegaConf.create(
        {
            "model_path": str(model_path.resolve()),
            "with_position": with_position,
            "with_color": with_color,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "dbscan": {
                "use": use_dbscan_denoising,
                "eps": dbscan_eps,
                "min_samples": dbscan_min_samples,
                "min_cluster_size": dbscan_min_cluster_size,
            },
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cluster_cfg, output_dir / "config.yaml", resolve=True)

    G = gaussians

    stats: dict = {"num_gaussians": G.num_points}
    mask_level = cfg.model.mask_level
    cluster_features = get_cluster_features(
        G,
        cfg.model,
        with_position,
        with_color,
    )
    print("\nMask level:", mask_level, f"({cluster_features.shape[-1]} dims)")
    
    # Try GPU HDBSCAN first
    try:
        if GPU_HDBSCAN is not None:
            print("Attempting GPU HDBSCAN...")
            labels: npt.NDArray = GPU_HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
            ).fit_predict(cluster_features)
        else:
            raise ImportError("GPU HDBSCAN not available")
    except Exception as e:
        # Check if it's the specific Pascal architecture error
        if "cudaErrorNoKernelImageForDevice" in str(e) or isinstance(e, ImportError):
            print("\n!!! CUDA Error (Pascal 6.1 detected) !!!")
            print("cuML does not support HDBSCAN on this architecture.")
            print("Falling back to CPU HDBSCAN (this might take a few minutes)...")
            labels: npt.NDArray = cpu_hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                core_dist_n_jobs=-1 # Use all CPU cores
            ).fit_predict(cluster_features)
        else:
            raise e

    unique, count = np.unique(labels, return_counts=True)
    sort_index = np.argsort(count)[::-1]
    unique = unique[sort_index]
    noisy_samples = int(np.sum(labels == -1))
    stats.update(
        {
            "level": mask_level,
            "num_clusters": len(unique),
            "noisy_samples": noisy_samples,
            "noise": (noisy_samples / len(labels)),
        }
    )
    print_cluster_info(labels)

    if use_dbscan_denoising:
        print(
            f"DBSCAN denoising: min_samples={dbscan_min_samples}, eps={dbscan_eps}, min_cluster_size={dbscan_min_cluster_size}"
        )
        labels = dbscan_denoising(
            G.get_xyz.detach().cpu().numpy(),
            labels,
            dbscan_min_samples,
            dbscan_eps,
            dbscan_min_cluster_size,
        )
        print_cluster_info(labels)
        stats["denoising"] = {
            "num_clusters": len(np.unique(labels)),
            "noisy_samples": int(np.sum(labels == -1)),
            "noise": (np.sum(labels == -1) / len(labels)),
        }

    np.save(output_dir / "labels.npy", labels)
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    print("Clustering can be found in:", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the model directory or to the parent when the latest model should be used",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory. If not provided, the output will be saved in a 'clustering' directory in the model directory.",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=0.0,
        help="Include position for clustering with weighting",
    )
    parser.add_argument(
        "--color",
        type=float,
        default=0.0,
        help="Include color for clustering with weighting",
    )
    parser.add_argument("--min-size", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--min-samples", type=int, default=None, help="Minimum samples")
    parser.add_argument(
        "--eps", type=float, default=0.0, help="Cluster selection epsilon"
    )
    parser.add_argument(
        "--dbscan-denoising", action="store_true", help="DBSCAN denoising"
    )
    parser.add_argument(
        "--dbscan-min-samples", type=int, default=5, help="DBSCAN minimum samples"
    )
    parser.add_argument("--dbscan-eps", type=float, default=0.5, help="DBSCAN epsilon")
    parser.add_argument("--dbscan-min-cluster-size", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.yaml").exists():
        model_dir = get_latest_model(model_dir)
        if not (model_dir / "config.yaml").exists():
            print("Could not find latest model.")
            exit()
        else:
            print(f"Using latest model: {model_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else model_dir / "clustering"
    )
    hdbscan_clustering(
        model_dir,
        output_dir,
        args.position,
        args.color,
        args.min_size,
        args.min_samples,
        args.eps,
        args.dbscan_denoising,
        args.dbscan_min_samples,
        args.dbscan_eps,
        args.dbscan_min_cluster_size,
    )
