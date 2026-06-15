"""
Preprocess ScanNet++ such that it can be used for our splatting pipeline.
The script creates a scene directory in output_dir for each scene in the split.
Link the images (dslr/undistorted_images) and transforms (dslr/nerfstudio/transforms_undistorted.json) files from the ScanNet++ directory per scene.
If pth_dir is provided, extract a point cloud from the .pth file and save it as a .ply file, similar to the ScanNet++ viz script.
"""

import shutil
from pathlib import Path

import open3d as o3d
import torch
from tqdm import tqdm


def main(
    scannetpp_dir: Path,
    output_dir: Path,
    split: str,
    pth_dir: Path | None,
    use_sampled: bool,
    copy: bool = False,
):
    data_dir = scannetpp_dir / "data"
    assert data_dir.exists(), f"Expected data directory at {data_dir}"

    prop_type = "sampled_" if use_sampled else "vtx_"

    scene_list = scannetpp_dir / "splits" / f"{split}.txt"
    assert scene_list.exists(), f"Expected split file at {scene_list}"
    scene_list = sorted(scene_list.read_text().strip().split("\n"))

    output_dir.mkdir(exist_ok=True, parents=True)

    for scene_id in tqdm(scene_list, total=len(scene_list), desc="Processing scenes"):
        assert (data_dir / scene_id).is_dir(), f"Expected directory for {scene_id}"
        scene_out_dir = output_dir / scene_id
        scene_out_dir.mkdir(exist_ok=True, parents=True)

        # images folder
        image_dir = data_dir / scene_id / "dslr" / "undistorted_images"
        if image_dir.exists():
            if copy:
                shutil.copytree(image_dir, scene_out_dir / "images", dirs_exist_ok=True)
            else:
                (scene_out_dir / "images").symlink_to(image_dir)

        # transformations
        transforms_file = (
            data_dir / scene_id / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        )
        if transforms_file.exists():
            if copy:
                shutil.copy2(transforms_file, scene_out_dir / "transforms.json")
            else:
                (scene_out_dir / "transforms.json").symlink_to(transforms_file)

        # points3d.ply
        if pth_dir is not None:
            pth_path = pth_dir / f"{scene_id}.pth"
            assert pth_path.exists(), f"Expected .pth file for {scene_id}"
            pth_data = torch.load(pth_path)
            vtx = pth_data[f"{prop_type}coords"]
            vtx_color = pth_data[f"{prop_type}colors"]
            out_fname = "points3d.ply"
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(vtx)
            pc.colors = o3d.utility.Vector3dVector(vtx_color)
            pc.estimate_normals()
            o3d.io.write_point_cloud(str(scene_out_dir / out_fname), pc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scannetpp_dir",
        type=Path,
        help="Path to the ScanNet++ directory where /data /splits etc. is located.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to the output directory where the data will be copied to.",
    )
    parser.add_argument("split", type=str, help="Name of the split to use.")
    parser.add_argument(
        "--pth-dir",
        type=Path,
        help="Path to the directory where the .pth files are located.",
    )
    parser.add_argument(
        "--use-sampled",
        action="store_true",
        help="Use sampled coordinates instead of vertex coordinates.",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy images instead of linking."
    )

    args = parser.parse_args()

    main(
        Path(args.scannetpp_dir),
        Path(args.output_dir),
        args.split,
        Path(args.pth_dir) if args.pth_dir is not None else None,
        args.use_sampled,
        args.copy,
    )
