from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from opensplat3d.gaussian_renderer import render
from opensplat3d.language import LanguageModel
from opensplat3d.language.utils import (
    CropParams,
    RenderParams,
    masks_to_crops,
    seg_pad_resize_masq,
)
from opensplat3d.params import PipeParams
from opensplat3d.utils.setup_utils import SetupParams, setup


@dataclass
class Stats:
    cam_idx: int
    pred_mask: np.ndarray
    area: int
    label_count: int
    visible_count: int


def compute_ious(gt_mask: torch.Tensor, pred_mask: torch.Tensor):
    ious = []
    gt_labels = torch.unique(gt_mask)
    gt_labels = gt_labels[gt_labels != -1]
    for label in gt_labels:
        intersection = (gt_mask == label) & pred_mask
        union = (gt_mask == label) | pred_mask
        iou = intersection.sum() / union.sum()
        ious.append(iou)
    return ious


def compute_view_stats(
    render_params: RenderParams,
    mask_color: torch.Tensor,
    pred_threshold: float,
    label_mask: torch.Tensor,
    label_count: int,
):
    stats: list[Stats] = []
    for i, cam in enumerate(render_params.cameras):
        render_pkg = render(
            cam,
            render_params.gaussians,
            render_params.pipe_params,
            render_params.bg,
            render_params.model_params.sh_degree,
            override_color=mask_color,
            render_features=False,
        )
        assert render_pkg.render is not None, "Rendered image is None"
        image = render_pkg.render.clamp(0, 1).permute(1, 2, 0).contiguous()
        pred_mask = image.mean(dim=-1) > pred_threshold
        area = pred_mask.flatten().sum()

        visible_count = (render_pkg.visibility_filter.cpu() & label_mask).sum()

        stats.append(
            Stats(
                i,
                pred_mask.cpu().numpy(),
                int(area.cpu().item()),
                label_count,
                int(visible_count.item()),
            )
        )
    return stats


def preprocess_crops(
    render_params: RenderParams,
    stats: list[Stats],
    lang_model: LanguageModel,
    crop_params: CropParams,
    use_rendering: bool,
):
    all_crops: dict[str, tuple[list[torch.Tensor], list[torch.Tensor] | None]] = {}
    for stat in stats:
        pred_mask = torch.from_numpy(stat.pred_mask)
        if use_rendering:
            render_pkg = render(
                render_params.cameras[stat.cam_idx],
                render_params.gaussians,
                render_params.pipe_params,
                render_params.bg,
                render_params.model_params.sh_degree,
                render_features=False,
            )
            assert render_pkg.render is not None, "Rendered image is None"
            image = render_pkg.render.clamp(0, 1).mul(255).to(dtype=torch.uint8).cpu()
        else:
            image = (
                render_params.cameras[stat.cam_idx]
                .original_image.clamp(0, 1)
                .mul(255)
                .to(dtype=torch.uint8)
            )

        old_expansion_ration = crop_params.expansion_ratio
        area_ratio = stat.area / (image.shape[1] * image.shape[2])
        if crop_params.dynamic_ratio:
            expansion_ratio = (
                crop_params.expansion_ratio if area_ratio < 0.0075 else 0.1
            )
        else:
            expansion_ratio = crop_params.expansion_ratio

        crop_params.expansion_ratio = expansion_ratio

        crops: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        if crop_params.levels == 0:
            assert lang_model.model_type == "masqclip", (
                "Non-crop mode only available for MasQClip"
            )
            crops_def, crop_masks_def = seg_pad_resize_masq(
                image,
                pred_mask.unsqueeze(0),
                lang_model.img_size,
            )
            crops["default"] = (
                lang_model.preprocess_images(
                    crops_def.permute(0, 2, 3, 1).contiguous().numpy()
                ),
                crop_masks_def,
            )
        else:
            crops_def, crop_masks_def, _, _ = masks_to_crops(
                image,
                pred_mask.unsqueeze(0),
                crop_params,
                lang_model,
            )
            crops_def = crops_def.squeeze(0)
            crop_masks_def = crop_masks_def.squeeze(0).unsqueeze(1)
            crops["default"] = (
                crops_def,
                crop_masks_def if lang_model.model_type == "masqclip" else None,
            )

        crop_params.expansion_ratio = old_expansion_ration

        for name, crops_ in crops.items():
            if name not in all_crops:
                all_crops[name] = ([], [])
            all_crops[name][0].append(crops_[0])
            if crops_[1] is not None:
                x = all_crops[name][1]
                if x is not None:
                    x.append(crops_[1])

        for name, crops_ in all_crops.items():
            if crops_[1] is not None and len(crops_[1]) == 0:
                all_crops[name] = (crops_[0], None)

    return {
        name: (
            torch.cat(crops_[0]),
            torch.cat(crops_[1]) if crops_[1] is not None else None,
        )
        for name, crops_ in all_crops.items()
    }


def compute_embeddings(
    setup_params: SetupParams,
    lang_model: LanguageModel,
    topk: int,
    crop_params: CropParams,
    use_rendering: bool,
    pred_threshold: float,
) -> dict:
    model_path = Path(setup_params.model_params.model_path)
    cameras = setup_params.scene.get_train_cameras()

    pipe_params = PipeParams()
    bg_color = [1, 1, 1] if setup_params.model_params.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device=setup_params.device)

    assert len(cameras) >= topk, "Not enough cameras for topk"

    labels = torch.from_numpy(np.load(model_path / "clustering" / "labels.npy"))

    render_params = RenderParams(
        setup_params.gaussians,
        setup_params.model_params,
        cameras,
        pipe_params,
        bg,
    )

    mask_level = setup_params.config.model.mask_level
    assert labels.shape[0] == setup_params.gaussians.num_points, "labels shape mismatch"

    unique_labels: torch.Tensor = labels.unique()
    unique_labels = unique_labels[unique_labels != -1]

    print(f"\nMask level: {setup_params.model_params.mask_level}")
    print(f"Num. of instances: {len(unique_labels)}")

    lang_img_embeds: list[torch.Tensor] = []
    out_valid: list[bool] = []
    for label in tqdm(unique_labels, desc=mask_level, total=unique_labels.shape[0]):
        label_mask = labels == label
        # set mask gaussian color to white and all other to black
        mask_color = torch.tensor([[1.0, 1.0, 1.0]], device=setup_params.device).repeat(
            labels.shape[0], 1
        )
        mask_color[~label_mask] = 0

        label_count = int(label_mask.sum().item())

        # compute predicted mask of instance per view
        stats = compute_view_stats(
            render_params,
            mask_color,
            pred_threshold,
            label_mask,
            label_count,
        )

        max_area = max([x.area for x in stats])
        if max_area == 0:
            lang_img_embeds.append(
                torch.zeros(lang_model.embed_dim, dtype=lang_model.dtype)
            )
            out_valid.append(False)
            continue

        # visibility score sorting
        stats = sorted(
            stats,
            key=lambda x: (x.visible_count / x.label_count) * (x.area / max_area),
            reverse=True,
        )[:topk]

        # render topk crops and apply CLIP preprocessing
        crops = preprocess_crops(
            render_params,
            stats,
            lang_model,
            crop_params,
            use_rendering,
        )
        crops = {
            name: (
                crop[0].to(lang_model.dtype).cuda(),
                crop[1].to(lang_model.dtype).cuda() if crop[1] is not None else None,
            )
            for name, crop in crops.items()
        }
        # compute language embeddings
        with torch.no_grad():
            embeddings = {
                name: lang_model.embed_images(crop[0], crop[1], normalize=False)
                for name, crop in crops.items()
            }
            if lang_model.model_type == "masqclip":
                # remove query dimension
                embeddings = {
                    name: embeds.squeeze(1) for name, embeds in embeddings.items()
                }
            merged_embeddings = embeddings["default"]
            merged_embeddings = merged_embeddings / merged_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )
            # mean over all topk crops and all clip levels
            lang_img_embeds.append(merged_embeddings.mean(dim=0).cpu())
        out_valid.append(True)

    assert len(lang_img_embeds) == len(out_valid), "Embeddings and valid mismatch"
    return {
        "embeddings": torch.stack(lang_img_embeds).numpy(),
        "valid": np.array(out_valid),
    }


def embed(
    model_path: Path,
    lang_model_type: str,
    topk: int,
    levels: int,
    masked: bool,
    ratio: float,
    dynamic_ratio: bool,
    alpha_blend: float,
    use_rendering: bool,
    pred_threshold: float,
):
    setup_params = setup(model_path)

    lang_model = LanguageModel(lang_model_type)
    crop_params = CropParams(
        lang_model.img_size,
        levels,
        masked,
        ratio,
        dynamic_ratio,
        alpha_blend,
    )

    embeddings = compute_embeddings(
        setup_params,
        lang_model,
        topk,
        crop_params,
        use_rendering,
        pred_threshold,
    )

    print(f"\nInvalid embeddings:  {(~embeddings['valid']).sum()}")

    embed_config = {
        "lang_model": lang_model.model_type,
        "topk": topk,
        "levels": levels,
        "masked": masked,
        "ratio": ratio,
        "dynamic_ratio": dynamic_ratio,
        "alpha_blend": alpha_blend,
        "use_rendering": use_rendering,
        "pred_threshold": pred_threshold,
    }
    embeddings["config"] = embed_config

    model_path = Path(setup_params.model_params.model_path)
    output_file = model_path / f"{lang_model.model_type}_embeddings.pth"
    torch.save(embeddings, output_file)
    print(f"\nEmbeddings saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Path to the model directory")
    parser.add_argument(
        "--lang-model",
        type=str,
        choices=["clip", "siglip", "masqclip"],
        default="masqclip",
        help="Language model",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of topk crops")
    parser.add_argument("--levels", type=int, default=3, help="Number of crop levels")
    parser.add_argument(
        "--ratio", type=float, default=0.3, help="Crop level expansion ratio"
    )
    parser.add_argument(
        "--dynamic-ratio",
        action="store_true",
        help="Dynamic clip level expansion ratio",
    )
    parser.add_argument(
        "--masked", action="store_true", help="Use masked crops for embeddings"
    )
    parser.add_argument(
        "--alpha-blend", type=float, default=0.0, help="Alpha blend for masking"
    )
    parser.add_argument(
        "--rendering", action="store_true", help="Use rendering for crops"
    )
    parser.add_argument(
        "--pred-thresh", type=float, default=0.2, help="Predicted mask threshold"
    )

    args = parser.parse_args()

    print(f"model_dir: {args.model_dir}")
    print(f"lang model: {args.lang_model}")
    print(f"topk: {args.topk}")
    print(f"crop levels: {args.levels}")
    print(f"masked: {args.masked}")
    print(f"expansion ratio: {args.ratio}")
    print(f"dynamic ratio: {args.dynamic_ratio}")
    print(f"alpha blend: {args.alpha_blend}")
    print(f"rendering: {args.rendering}")
    print(f"prediction threshold: {args.pred_thresh}")
    print()

    embed(
        Path(args.model_dir),
        args.lang_model,
        args.topk,
        args.levels,
        args.masked,
        args.ratio,
        args.dynamic_ratio,
        args.alpha_blend,
        args.rendering,
        args.pred_thresh,
    )
