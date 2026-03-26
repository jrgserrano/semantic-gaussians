import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from opensplat3d.language.embed import compute_view_stats
from opensplat3d.language.utils import RenderParams
from opensplat3d.params import PipeParams
from opensplat3d.utils.setup_utils import setup


def plot_clusters(model_dir: Path, pred_threshold: float):
    setup_params = setup(model_dir)

    output_dir = model_dir / "clustering"
    labels_path = output_dir / "labels.npy"

    if not labels_path.exists():
        print(f"No clustering found at {labels_path}")
        return

    labels = torch.from_numpy(np.load(labels_path))
    unique_labels = labels.unique()
    unique_labels = unique_labels[unique_labels != -1]

    cameras = setup_params.scene.get_train_cameras()

    pipe_params = PipeParams()
    bg_color = [1, 1, 1] if setup_params.model_params.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device=setup_params.device)

    render_params = RenderParams(
        setup_params.gaussians,
        setup_params.model_params,
        cameras,
        pipe_params,
        bg,
    )

    print(f"Evaluating {len(unique_labels)} clusters for plotting...")

    appearances_list = []
    avg_scores_list = []
    cluster_ids = []

    for label in tqdm(unique_labels, desc="Evaluating clusters", total=unique_labels.shape[0]):
        label_id = int(label.item())
        label_mask = labels == label

        mask_color = torch.tensor([[1.0, 1.0, 1.0]], device=setup_params.device).repeat(
            labels.shape[0], 1
        )
        mask_color[~label_mask] = 0

        label_count = int(label_mask.sum().item())

        stats = compute_view_stats(
            render_params,
            mask_color,
            pred_threshold,
            label_mask,
            label_count,
        )

        max_area = max([x.area for x in stats]) if stats else 0

        appearances = 0
        total_score = 0.0

        if max_area > 0:
            for stat in stats:
                if stat.area > 0:
                    appearances += 1
                    score = (stat.visible_count / stat.label_count) * (stat.area / max_area)
                    total_score += score

        avg_score = total_score / appearances if appearances > 0 else 0.0

        appearances_list.append(appearances)
        avg_scores_list.append(avg_score)
        cluster_ids.append(str(label_id))

    # Plot 1: Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(appearances_list, avg_scores_list, alpha=0.7, edgecolors='k')

    plt.title("Cluster Visualization Quality")
    plt.xlabel("Number of Appearances (Views with Area > 0)")
    plt.ylabel("Average Visualization Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Optional styling to help visualize common thresholds
    plt.axvline(x=5, color='r', linestyle='--', alpha=0.4, label='x=5 (Typical Min Appearances)')
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.4, label='y=0.05 (Typical Min Score)')
    plt.legend()

    plot_path = output_dir / "cluster_scores_scatter.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Bar Chart for Appearances per Object
    plt.figure(figsize=(max(10, len(cluster_ids) * 0.3), 6))
    plt.bar(cluster_ids, appearances_list, color='skyblue', edgecolor='k')
    plt.title("Number of Appearances per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Appearances")
    plt.xticks(rotation=90 if len(cluster_ids) > 15 else 0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.4, label='Min Appearances = 5')
    plt.legend()
    bar_plot_path = output_dir / "cluster_appearances_bar.png"
    plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Bar Chart for Scores per Object
    plt.figure(figsize=(max(10, len(cluster_ids) * 0.3), 6))
    plt.bar(cluster_ids, avg_scores_list, color='lightgreen', edgecolor='k')
    plt.title("Average Visualization Score per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Average Score")
    plt.xticks(rotation=90 if len(cluster_ids) > 15 else 0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.4, label='Min Score = 0.05')
    plt.legend()
    score_bar_plot_path = output_dir / "cluster_scores_bar.png"
    plt.savefig(score_bar_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved scatter plot to: {plot_path}")
    print(f"Saved appearances bar chart to: {bar_plot_path}")
    print(f"Saved score bar chart to: {score_bar_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Path to the model directory")
    parser.add_argument("--pred-thresh", type=float, default=0.2, help="Predicted mask threshold")

    args = parser.parse_args()

    plot_clusters(args.model_dir, args.pred_thresh)
