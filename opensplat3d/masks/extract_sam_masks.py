from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
from omegaconf import OmegaConf
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    SAM_VERSION = 2
except ImportError:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    SAM_VERSION = 1
from tqdm import tqdm

from opensplat3d.masks.sam_levels_model import SamLevelsAutomaticMaskGenerator
from opensplat3d.masks.utils import masks_update

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def process_mask_level(
    image: npt.NDArray,
    masks: list[dict[str, Any]],
    postprocess: str | None,
    binary_mask: bool = False,
    sort_key: str | None = None,
    pre_kernel_size: tuple[int, int] | None = None,
    post_kernel_size: tuple[int, int] | None = None,
):
    if postprocess == "lang-splat":
        masks = masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    if sort_key is not None:
        assert sort_key in {
            "predicted_iou",
            "stability_score",
            "score",
            "area",
            "area+score",
        }
        if sort_key == "score":
            masks = sorted(
                masks, key=lambda x: x["predicted_iou"] * x["stability_score"]
            )
        elif sort_key == "area":
            masks = sorted(
                masks, key=lambda x: x["area"], reverse=True
            )  # prioritize smaller masks
        elif sort_key == "area+score":
            masks = sorted(
                masks, key=lambda x: x["predicted_iou"] * x["stability_score"]
            )
            masks = sorted(
                masks, key=lambda x: x["area"], reverse=True
            )  # prioritize smaller masks
        else:
            masks = sorted(masks, key=lambda x: x[sort_key])
    if binary_mask:
        if pre_kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, pre_kernel_size)
            for m in masks:
                mask: npt.NDArray = m["segmentation"]
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask.astype(bool)
                m["segmentation"] = mask
        return np.stack([m["segmentation"] for m in masks])
    else:
        fullmask = -1 * np.ones(image.shape[:2], dtype=np.int32)

        kernel = None
        if pre_kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, pre_kernel_size)

        for i, m in enumerate(masks):
            mask: npt.NDArray = m["segmentation"]
            if kernel is not None:
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask.astype(bool)
            fullmask[mask] = i

        if post_kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, post_kernel_size)
            for mask_id in np.unique(fullmask):
                mask = fullmask == mask_id
                modified_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                modified_mask = cv2.dilate(modified_mask, kernel, iterations=1)
                modified_mask = np.asarray(modified_mask, dtype=bool)
                # Combine the modified mask back into the fullmask
                fullmask[mask ^ modified_mask] = -1

        # Ensure that the labels are contiguous
        idx = np.unique(fullmask)
        idx = idx[idx >= 0]  # ignore -1 label
        for i, j in enumerate(idx):
            fullmask[fullmask == j] = i

        return fullmask


def compute_masks(
    mask_generator: Any,
    image_path: Path,
    postprocess: str | None,
    binary_mask: bool = False,
    sort_key: str | None = None,
    pre_kernel_size: tuple[int, int] | None = None,
    post_kernel_size: tuple[int, int] | None = None,
):
    image = iio.imread(image_path)
    if isinstance(mask_generator, SamLevelsAutomaticMaskGenerator):
        mask_levels = mask_generator.generate(image)
        # NOTE: the order of mask_levels of SAM is [default, subpart, part, whole]
        levels = ["default", "subpart", "part", "whole"]
        assert len(mask_levels) == len(levels), f"Expected 4 levels of masks {levels}"
        output: dict[str, npt.NDArray] = {}
        for masks, level in zip(mask_levels, levels):
            output[level] = process_mask_level(
                image,
                masks,
                postprocess,
                binary_mask,
                sort_key,
                pre_kernel_size,
                post_kernel_size,
            )
        return output
    else:
        masks = mask_generator.generate(image)
        return {
            "default": process_mask_level(
                image,
                masks,
                postprocess,
                binary_mask,
                sort_key,
                pre_kernel_size,
                post_kernel_size,
            )
        }


def main(
    scene_dir: Path,
    output_dir: Path,
    image_subdir: str = "images",
    levels: bool = False,
    postprocess: str | None = None,
    binary_mask: bool = False,
    sort_key: str | None = None,
    pre_kernel_size: tuple[int, int] | None = None,
    post_kernel_size: tuple[int, int] | None = None,
    min_mask_region_area: int = 0,
    compress: bool = False,
    nth_frames: int = 1,
):
    if not scene_dir.exists():
        print(f"Scene {scene_dir} does not exist")
        return
    if not scene_dir.is_dir():
        print(f"Scene {scene_dir} is not a directory")
        return

    image_dir = scene_dir / image_subdir
    if not image_dir.exists():
        if (scene_dir / "traj.txt").exists() and (scene_dir / "results").exists():
            print("traj.txt found, automatically switching image_subdir to 'results' for Replica dataset.")
            image_dir = scene_dir / "results"
            image_subdir = "results"
        else:
            print(f"Image directory {image_dir} does not exist")
            return
    if not image_dir.is_dir():
        print(f"Image directory {image_dir} is not a directory")
        return

    print("Loading SAM 2 model")
    sam2_checkpoint = "ckpts/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l"
    sam = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    if levels:
        # Note: levels logic might need update in sam_levels_model.py
        mask_generator = SamLevelsAutomaticMaskGenerator(
            sam, min_mask_region_area=min_mask_region_area
        )
    else:
        mask_generator = SAM2AutomaticMaskGenerator(
            sam, min_mask_region_area=min_mask_region_area
        )

    output_dir.mkdir(exist_ok=True, parents=True)
    config = OmegaConf.create(
        {
            "scene_dir": str(scene_dir.resolve()),
            "image_subdir": image_subdir,
            "levels": levels,
            "postprocess": postprocess,
            "binary_mask": binary_mask,
            "sort_key": sort_key,
            "pre_kernel_size": pre_kernel_size,
            "post_kernel_size": post_kernel_size,
            "min_mask_region_area": min_mask_region_area,
            "compress": compress,
        }
    )
    OmegaConf.save(config, output_dir / "config.yaml", resolve=True)

    imgs = sorted(
        [
            x
            for x in image_dir.iterdir()
            if x.is_file()
            and x.suffix.lower() in IMAGE_SUFFIXES
            and not x.name.startswith("depth")   # Replica: depth0000.png
            and ".depth." not in x.name           # Polycam: 0.depth.png
        ]
    )
    if nth_frames > 1:
        imgs = imgs[::nth_frames]
        print(f"Subsampling masks frame extraction with nth_frames={nth_frames}. Selected {len(imgs)} frames.")
    for image_path in tqdm(imgs, total=len(imgs)):
        mask_path = output_dir / f"{image_path.stem}.npz"
        if mask_path.exists():
            continue
        mask_levels = compute_masks(
            mask_generator,
            image_path,
            postprocess,
            binary_mask,
            sort_key,
            pre_kernel_size,
            post_kernel_size,
        )
        for level, masks in mask_levels.items():
            assert not binary_mask or masks.ndim == 3, (
                f"Expected masks of shape (N, H, W), got {masks.ndim} for level {level}"
            )

        if compress:
            np.savez_compressed(mask_path, **mask_levels)
        else:
            np.savez(mask_path, **mask_levels)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Extract SAM masks for a scene")
    parser.add_argument(
        "source_dir", type=str, help="Scene directory to extract SAM masks for"
    )
    parser.add_argument(
        "--output-subdir", type=str, default="sam", help="Output subdirectory for masks"
    )
    parser.add_argument(
        "--img-subdir",
        type=str,
        default="images",
        help="Subdirectory of scene directory containing images",
    )
    parser.add_argument(
        "--levels",
        action="store_true",
        help="Extract all SAM levels instead of just the default SAM output",
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        choices=["lang-splat"],
        help="Use mask postprocessing of lang-splat",
    )
    parser.add_argument(
        "--binary-mask",
        action="store_true",
        help="Output binary masks instead of merged masks",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["predicted_iou", "stability_score", "score", "area", "area+score"],
        help="Sort masks by the given key",
    )
    parser.add_argument(
        "--pre-kernel-size",
        type=int,
        help="Size of morphological kernel for preprocessing",
    )
    parser.add_argument(
        "--post-kernel-size",
        type=int,
        help="Size of morphological kernel for postprocessing",
    )
    parser.add_argument(
        "--min-mask-region-area",
        type=int,
        default=0,
        help="Minimum area of mask region to consider",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the masks",
    )
    parser.add_argument(
        "--nth-frames",
        type=int,
        default=1,
        help="Only extract masks for every Nth frame to save time (e.g. 5 means 1/5th of frames)",
    )

    args = parser.parse_args()

    source_path = Path(args.source_dir)
    output_path: Path = source_path / args.output_subdir

    main(
        source_path,
        output_path,
        args.img_subdir,
        args.levels,
        args.postprocess,
        args.binary_mask,
        args.sort,
        (args.pre_kernel_size, args.pre_kernel_size) if args.pre_kernel_size else None,
        (args.post_kernel_size, args.post_kernel_size)
        if args.post_kernel_size
        else None,
        args.min_mask_region_area,
        args.compress,
        args.nth_frames,
    )
