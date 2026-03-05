from pathlib import Path
from typing import Any

import torch

from opensplat3d.data.blender_reader import BlenderReader
from opensplat3d.data.colmap_loader import read_points3D_binary, read_points3D_text
from opensplat3d.data.colmap_reader import ColmapReader
from opensplat3d.data.nerfstudio_reader import NerfStudioReader
from opensplat3d.data.reader import Reader
from opensplat3d.params import ModelParams
from opensplat3d.utils.scene_utils import (
    BasicPointCloud,
    SceneInfo,
    fetch_ply,
    generate_random_point_cloud,
    get_nerf_pp_norm,
    store_ply,
)


def read_scene_info(
    reader: Reader[Any],
    path: Path,
    eval: bool,
    num_pts: int = 100000,
    init_type: str = "sample",
    progbar: bool = False,
):
    train_cam_infos = reader.load_train(progbar)
    test_cam_infos = reader.load_test(progbar)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    print(f"Train: {len(train_cam_infos)} | Test: {len(test_cam_infos)}")

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    if (path / "sparse").exists():
        ply_path = path / "sparse" / "0" / "points3D.ply"

        if not ply_path.exists():
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(ply_path.with_suffix(".bin"))
            except Exception:
                xyz, rgb, _ = read_points3D_text(ply_path.with_suffix(".txt"))
            store_ply(ply_path, xyz, rgb)
    else:
        ply_path = path / "points3d.ply"
        if not ply_path.exists():
            # Since only COLMAP datasets have initialization points, we start with random points
            pcd = generate_random_point_cloud(
                ply_path,
                num_pts,
                nerf_normalization.radius,
                -nerf_normalization.translate,
            )

    is_random = False
    try:
        pcd, is_random = fetch_ply(ply_path)
        if not is_random and init_type == "random":
            # Generate random points but don´t actually override the existing point cloud which is potentially from SfM
            print("Overwriting existing point cloud with random points without saving.")
            pcd = generate_random_point_cloud(
                ply_path,
                num_pts,
                nerf_normalization.radius,
                -nerf_normalization.translate,
                save=False,
            )
        if init_type == "sample":
            num_pts_pcd = pcd.points.shape[0]
            if num_pts < num_pts_pcd:
                rand_indices = torch.randperm(num_pts_pcd)[:num_pts]
                pcd = BasicPointCloud(
                    points=pcd.points[rand_indices],
                    colors=pcd.colors[rand_indices],
                    normals=pcd.normals[rand_indices],
                )
            print(
                f"Sampled {pcd.points.shape[0]}/{num_pts_pcd} points from the point cloud."
            )
    except Exception:
        pcd = None

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_pcd_random=is_random or init_type == "random",
    )


def load_scene_info(model_params: ModelParams, progbar: bool = True) -> SceneInfo:
    source_path = Path(model_params.source_path)
    reader: Reader | None = None
    kwargs: dict[str, Any] = dict(
        mask_subdir=model_params.mask_subdir,
        mask_level=model_params.mask_level,
        num_frames=model_params.num_frames,
        nth_frames=model_params.nth_frames,
        frames_dist=model_params.frames_dist,
    )
    if (source_path / "sparse").exists():
        reader = ColmapReader(
            source_path,
            model_params.images,
            model_params.test_hold,
            **kwargs,
        )
    elif (source_path / "transforms_train.json").exists():
        print("Found transforms_train.json file, assuming Blender data set!")
        reader = BlenderReader(source_path, model_params.white_background, **kwargs)
    elif (source_path / "transforms.json").exists():
        print("Found transforms.json file, assuming NeRFStudio data set!")
        reader = NerfStudioReader(source_path, model_params.images, **kwargs)
    else:
        assert False, "Could not recognize scene type!"

    return read_scene_info(
        reader,
        source_path,
        model_params.eval,
        model_params.init_points,
        model_params.init_type,
        progbar=progbar,
    )
