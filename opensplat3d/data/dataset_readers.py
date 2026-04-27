from pathlib import Path
from typing import Any

import numpy as np
import torch

from opensplat3d.data.blender_reader import BlenderReader
from opensplat3d.data.colmap_loader import read_points3D_binary, read_points3D_text
from opensplat3d.data.colmap_reader import ColmapReader
from opensplat3d.data.nerfstudio_reader import NerfStudioReader
from opensplat3d.data.polycam_reader import PolycamReader
from opensplat3d.data.replica_reader import ReplicaReader
from opensplat3d.data.ros2_reader import ROS2Reader
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

    if path.is_file():
        ply_path = path.parent / f"{path.stem}_points3d.ply"
    elif (path / "sparse").exists():
        ply_path = path / "sparse" / "0" / "points3D.ply"
    else:
        ply_path = path / "points3d.ply"

    if not ply_path.exists():
        if (path / "sparse").exists():
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(ply_path.with_suffix(".bin"))
            except Exception:
                xyz, rgb, _ = read_points3D_text(ply_path.with_suffix(".txt"))
            store_ply(ply_path, xyz, rgb)
        else:
            # Since only COLMAP datasets have initialization points, we start with random points
            pcd = generate_random_point_cloud(
                ply_path,
                num_pts if num_pts > 0 else 100000,
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
        if init_type == "sample" and num_pts > 0:
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

        # Load SAM instance labels if available
        instances_path = path / "points3d_instances.npz"
        if instances_path.exists() and not is_random:
            try:
                inst_data = np.load(instances_path)
                inst_xyz = torch.from_numpy(inst_data["xyz"])   # (M, 3) – might differ from pcd
                inst_ids = torch.from_numpy(inst_data["instance_id"])  # (M,)
                # Match to the (possibly sampled) pcd by finding nearest neighbour in xyz
                # Fast: build a set from rounded coords for exact-match when pcd == points3d.ply
                pcd_np = pcd.points.numpy()
                inst_xyz_np = inst_xyz.numpy()
                # Round to 4 decimal places to handle float precision
                coord_to_id = {
                    tuple(np.round(row, 4)): int(iid)
                    for row, iid in zip(inst_xyz_np, inst_ids.numpy())
                }
                matched = np.array([
                    coord_to_id.get(tuple(np.round(p, 4)), -1) for p in pcd_np
                ], dtype=np.int32)
                pcd = BasicPointCloud(
                    points=pcd.points,
                    colors=pcd.colors,
                    normals=pcd.normals,
                    instance_ids=torch.from_numpy(matched),
                )
                n_labelled = (matched >= 0).sum()
                print(f"Instance labels loaded: {n_labelled}/{len(matched)} points labelled "
                      f"({len(np.unique(matched[matched>=0]))} unique instances).")
            except Exception as e:
                print(f"Warning: could not load instance labels from {instances_path}: {e}")
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
        import json
        with open(source_path / "transforms.json") as _f:
            _meta = json.load(_f)
        # NerfStudio/ScanNet++ format requires camera_model and separate test_frames.
        # Polycam / Record3D / NeRF-style uses a single 'frames' list without camera_model.
        if "camera_model" in _meta and "test_frames" in _meta:
            print("Found transforms.json with camera_model, assuming NeRFStudio (ScanNet++) data set!")
            reader = NerfStudioReader(source_path, model_params.images, **kwargs)
        else:
            print("Found transforms.json, assuming Polycam / NeRF-style data set!")
            reader = PolycamReader(
                source_path,
                test_hold=model_params.test_hold,
                mask_subdir=kwargs.get("mask_subdir"),
                mask_level=kwargs.get("mask_level", "default"),
                num_frames=kwargs.get("num_frames", -1),
                nth_frames=kwargs.get("nth_frames", -1),
                frames_dist=kwargs.get("frames_dist", "uniform"),
            )
    elif (source_path / "traj.txt").exists():
        print("Found traj.txt file, assuming Replica data set!")
        reader = ReplicaReader(
            source_path,
            test_hold=model_params.test_hold,
            num_frames=kwargs.get("num_frames", -1),
            nth_frames=kwargs.get("nth_frames", -1),
            frames_dist=kwargs.get("frames_dist", "uniform"),
            mask_subdir=kwargs.get("mask_subdir", None),
            mask_level=kwargs.get("mask_level", "default"),
        )
    elif source_path.suffix == ".db3" or (source_path / "astra_lab_0.db3").exists():
        print("Found ROS2 Bag, assuming ROS2 dataset!")
        bag_file = source_path if source_path.suffix == ".db3" else (source_path / "astra_lab_0.db3")
        
        # Define intrinsics from user input
        intrinsics = {
            "fx": 516.4535522460938,
            "fy": 516.4535522460938,
            "cx": 332.4849548339844,
            "cy": 242.23336791992188
        }
        
        reader = ROS2Reader(
            str(bag_file),
            world_frame=model_params.world_frame,
            camera_frame=model_params.camera_frame,
            intrinsics=intrinsics,
            num_frames=kwargs.get("num_frames", -1),
            nth_frames=kwargs.get("nth_frames", -1),
            mask_subdir=model_params.mask_subdir,
        )
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
