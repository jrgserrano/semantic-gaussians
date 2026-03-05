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

from opensplat3d.config import load_config, save_config, to_dict
from opensplat3d.gaussian_model import GaussianModel
from opensplat3d.gaussian_renderer import render
from opensplat3d.losses import l1_loss
from opensplat3d.params import PipeParams
from opensplat3d.scene import Camera, Scene
from opensplat3d.utils.metric_utils import psnr


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
            project=args.wandb if args.wandb != "" else "opensplat3d",
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
):
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
        for config in validation_configs:
            if len(config.cameras) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config.cameras):
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
                                f"{config.name}_view_{viewpoint.name}/render": [
                                    wandb.Image(image)
                                ],
                            },
                            step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            wandb.log(
                                {
                                    f"{config.name}_view_{viewpoint.name}/ground_truth": [
                                        wandb.Image(gt_image)
                                    ],
                                },
                                step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config.cameras)
                l1_test /= len(config.cameras)
                print(
                    f"\n[ITER {iteration}] Evaluating {config.name}: L1 {l1_test} PSNR {psnr_test}"
                )
        torch.cuda.empty_cache()
