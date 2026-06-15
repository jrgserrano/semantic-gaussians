import datetime
import os
import random
import sys
import uuid
from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

import torch
import wandb

from core.config import load_config, save_config, to_dict
from core.gaussian_model import GaussianModel
from core.gaussian_renderer import render
from core.losses import l1_loss, ssim
from core.params import PipeParams
from core.scene import Camera, Scene
from core.utils.metric_utils import psnr
from core.utils.loss_utils import get_mean_prototypes
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lpips

_lpips_vgg = None


def setup_training(args: Namespace):
    config = load_config(args.overrides, args.config)
    model_path = Path(config.model.model_path).resolve()

    if "SLURM_JOB_ID" in os.environ:
        uid = f"slurm{os.environ['SLURM_JOB_ID']}"
    else:
        uid = str(uuid.uuid4())[:8]
    dtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    model_path = model_path / f"{dtime}-{uid}"

    config.model.model_path = str(model_path)
    config.model.source_path = str(Path(config.model.source_path).resolve())

    model_path.mkdir(parents=True, exist_ok=True)

    save_config(config, args.overrides, model_path)

    if args.wandb is not None:
        name = str(model_path.relative_to(model_path.parent.parent))
        wandb.init(
            project=args.wandb if args.wandb != "" else "core",
            config=to_dict(config),
            dir=model_path,
            name=name,
            save_code=False,
        )

    # Write command line arguments to file
    with open(model_path / "command.txt", "w") as f:
        f.write(" ".join(sys.argv))

    return config


def train_cameras(scene: Scene):
    viewpoint_stack = scene.get_train_cameras().copy()
    while len(viewpoint_stack) > 0:
        camera = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        yield camera
        if len(viewpoint_stack) == 0:
            viewpoint_stack = scene.get_train_cameras().copy()


class FisherCameraSampler:
    """
    Priority-based camera sampler that uses Fisher Information to select 
    the next best view for training.
    """
    def __init__(self, cameras: list):
        self.cameras = cameras
        self.uids = [cam.uid for cam in cameras]
        self.uid_to_idx = {cam.uid: i for i, cam in enumerate(cameras)}
        # Start with uniform intensity
        self.scores = torch.ones(len(cameras), dtype=torch.float32)

    def sample(self) -> Camera:
        # Use weighted random sampling. Higher score = higher probability.
        # We use a soft-max style temperature to avoid ignoring easy views entirely
        probabilities = self.scores / self.scores.sum()
        idx = torch.multinomial(probabilities, 1).item()
        return self.cameras[idx]

    def update_score(self, uid: int, surprise_value: float):
        if uid in self.uid_to_idx:
            idx = self.uid_to_idx[uid]
            clamped_surprise = torch.clamp(torch.tensor(surprise_value), min=0.01, max=50.0)
            self.scores[idx] = 0.8 * self.scores[idx] + 0.2 * clamped_surprise
            # Ensure scores don't drop to zero
            self.scores[idx] = max(self.scores[idx], 0.01)

    def get_stats(self):
        return {
            "max_score": self.scores.max().item(),
            "min_score": self.scores.min().item(),
            "mean_score": self.scores.mean().item(),
        }


class ValidationConfig(NamedTuple):
    name: str
    cameras: list[Camera]


def training_report(
    iteration: int,
    elapsed: float | None,
    l1: torch.Tensor | None,
    losses: dict[str, dict],
    testing_iterations: list[int],
    gaussians: GaussianModel,
    scene: Scene,
    pipe_params: PipeParams,
    background: torch.Tensor,
    max_sh_degree: int,
    active_sh_degree: int,
    optimizer: any = None,
):
    report_metrics = {}
    if wandb.run is not None:
        log_dict = {
            "total_points": gaussians.num_points,
            "opacity_histogram": wandb.Histogram(
                gaussians.get_opacity.cpu().numpy()  # type: ignore
            ),
        }
        for loss_name, loss_dict in losses.items():
            if not loss_dict.get("log", False):
                continue
            value: torch.Tensor | None = loss_dict.get("value", None)
            if value is not None:
                log_dict[f"train_loss_patches/{loss_name}"] = value.item()
            timing: float | None = loss_dict.get("timing", None)
            if timing is not None:
                log_dict[f"{loss_name}_time"] = timing

        if elapsed is not None:
            log_dict["iter_time"] = elapsed
        if l1 is not None:
            log_dict["train_loss_patches/l1_loss"] = l1.item()

        if gaussians.get_features is not None:
            features_norm = gaussians.get_features.norm(dim=-1)
            log_dict["features/mean"] = features_norm.mean().item()
            log_dict["features/std"] = features_norm.std().item()

        wandb.log(
            log_dict,
            step=iteration,
        )

    log_images = l1 is not None

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            ValidationConfig("test", scene.get_test_cameras()),
            ValidationConfig(
                "train",
                [
                    scene.get_train_cameras()[idx % len(scene.get_train_cameras())]
                    for idx in range(5, 30, 5)
                ],
            ),
        )
        for val_config in validation_configs:
            if len(val_config.cameras) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                
                global _lpips_vgg
                if _lpips_vgg is None:
                    _lpips_vgg = lpips.LPIPS(net="vgg").cuda()
                for idx, viewpoint in enumerate(val_config.cameras):
                    render_pkg = render(
                        viewpoint,
                        gaussians,
                        pipe_params,
                        background,
                        max_sh_degree,
                        active_sh_degree,
                    )
                    assert render_pkg.render is not None, "Rendered image is None"
                    image = render_pkg.render.clamp(0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda().clamp(0.0, 1.0)
                    if wandb.run is not None and idx < 5 and log_images:
                        wandb.log(
                            {
                                f"{val_config.name}_view_{viewpoint.name}/render": [
                                    wandb.Image(image)
                                ],
                            },
                            step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            wandb.log(
                                {
                                    f"{val_config.name}_view_{viewpoint.name}/ground_truth": [
                                        wandb.Image(gt_image)
                                    ],
                                },
                                step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image[None], gt_image[None]).mean().double()
                    with torch.no_grad():
                        lpips_test += _lpips_vgg(image * 2 - 1, gt_image * 2 - 1).mean().double()
                psnr_test /= len(val_config.cameras)
                ssim_test /= len(val_config.cameras)
                l1_test /= len(val_config.cameras)
                lpips_test /= len(val_config.cameras)
                print(
                    f"\n[ITER {iteration}] Evaluating {val_config.name}: L1 {l1_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}"
                )
                report_metrics[f"psnr_{val_config.name}"] = psnr_test.item()
                report_metrics[f"ssim_{val_config.name}"] = ssim_test.item()
                report_metrics[f"l1_{val_config.name}"] = l1_test.item()
                report_metrics[f"lpips_{val_config.name}"] = lpips_test.item()
        torch.cuda.empty_cache()
    return report_metrics


def save_quality_heatmaps(
    iteration: int,
    model_path: Path,
    viewpoint_name: str,
    rendered_image: torch.Tensor,
    rendered_features: torch.Tensor,
    gt_image: torch.Tensor,
    gt_masks: torch.Tensor | None,
    mask_dim: int,
):
    """
    Saves RGB error and Semantic error heatmaps as images.
    """
    diag_dir = model_path / "diagnostics" / f"iter_{iteration}"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 1. RGB L1 Error Map
    rgb_error = (rendered_image - gt_image).abs().mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_error, cmap="magma")
    plt.colorbar(label="L1 Error")
    plt.title(f"RGB Error Map - {viewpoint_name}")
    plt.savefig(diag_dir / f"{viewpoint_name}_rgb_error.png")
    plt.close()

    # 2. Semantic Consistency Map (if masks are available)
    if gt_masks is not None and rendered_features is not None:
        with torch.no_grad():
            # Get mean prototypes for the rendered features using GT masks
            # Since get_mean_prototypes assumes labels start from 1 (ignore -1)
            # we need to handle labels correctly.
            try:
                prototypes, binary_gt, _ = get_mean_prototypes(rendered_features[:mask_dim], gt_masks)
                # binary_gt is (I, H*W)
                # prototypes is (I, C)
                # Reconstruct full error map
                C, H, W = rendered_features[:mask_dim].shape
                flat_features = rendered_features[:mask_dim].flatten(1).T # (H*W, C)
                
                # For each pixel, find its prototype distance
                # binary_gt has 1 where the pixel belongs to instance i
                # We can do this efficiently:
                # pixel_prototypes = binary_gt.T @ prototypes # (H*W, C)
                pixel_prototypes = torch.matmul(binary_gt.transpose(0, 1).float(), prototypes) # (H*W, C)
                
                # Error is L2 distance in feature space
                sem_error = (flat_features - pixel_prototypes).pow(2).sum(dim=-1).sqrt()
                sem_error_map = sem_error.reshape(H, W).cpu().numpy()

                plt.figure(figsize=(10, 8))
                plt.imshow(sem_error_map, cmap="jet")
                plt.colorbar(label="Feature L2 Distance")
                plt.title(f"Semantic Error Map - {viewpoint_name}")
                plt.savefig(diag_dir / f"{viewpoint_name}_semantic_error.png")
                plt.close()
            except Exception as e:
                print(f"[ WARNING ] Failed to generate semantic heatmap: {e}")
