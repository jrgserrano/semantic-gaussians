import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from core.language.embed import compute_view_stats
from core.language.utils import RenderParams
from core.params import PipeParams
from core.utils.setup_utils import setup


def clean_clusters(
    model_dir: Path,
    min_appearances: int,
    min_score: float,
    pred_threshold: float,
):
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

    print(f"Cleaning {len(unique_labels)} clusters...")

    cleaned_labels = labels.clone()
    num_removed = 0

    for label in tqdm(unique_labels, desc="Evaluating clusters", total=unique_labels.shape[0]):
        label_mask = labels == label

        # Set mask gaussian color to white and all other to black
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

        if max_area == 0:
            cleaned_labels[label_mask] = -1
            num_removed += 1
            continue

        appearances = 0
        total_score = 0.0

        for stat in stats:
            if stat.area > 0:
                appearances += 1
                score = (stat.visible_count / stat.label_count) * (stat.area / max_area)
                total_score += score

        avg_score = total_score / appearances if appearances > 0 else 0.0

        if appearances < min_appearances or avg_score < min_score:
            cleaned_labels[label_mask] = -1
            num_removed += 1

    print(f"\nRemoved {num_removed} clusters out of {len(unique_labels)}")

    cleaned_labels_path = output_dir / "labels_cleaned.npy"
    np.save(cleaned_labels_path, cleaned_labels.cpu().numpy())
    print(f"Saved cleaned labels to: {cleaned_labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Path to the model directory")
    parser.add_argument("--min-appearances", type=int, default=5, help="Minimum number of appearances in views")
    parser.add_argument("--min-score", type=float, default=0.05, help="Minimum average visualization score")
    parser.add_argument("--pred-thresh", type=float, default=0.2, help="Predicted mask threshold")

    args = parser.parse_args()

    clean_clusters(
        args.model_dir,
        args.min_appearances,
        args.min_score,
        args.pred_thresh,
    )
