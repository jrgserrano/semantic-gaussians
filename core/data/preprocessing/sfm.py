#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging
import os
import shutil
from pathlib import Path

from core.data.colmap_loader import read_extrinsics_binary, read_points3D_binary


def feature_pipeline(
    source_path: Path,
    database_path: Path,
    colmap_command: str,
    camera: str,
    use_gpu: int,
):
    (source_path / "distorted" / "sparse").mkdir(exist_ok=True, parents=True)

    ## Feature extraction
    feat_extracton_cmd = " ".join(
        [
            colmap_command,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(source_path / "input"),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            camera,
            "--SiftExtraction.use_gpu",
            str(use_gpu),
        ]
    )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = " ".join(
        [
            colmap_command,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            str(use_gpu),
        ]
    )
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)


def colmap(source_path: Path, database_path: Path, colmap_command: str):
    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = " ".join(
        [
            colmap_command,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(source_path / "input"),
            "--output_path",
            str(source_path / "distorted" / "sparse"),
            "--Mapper.ba_global_function_tolerance=0.000001",
        ]
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)


def glomap(source_path: Path, database_path: Path, glomap_command: str):
    ### Bundle adjustment
    mapper_cmd = " ".join(
        [
            glomap_command,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(source_path / "input"),
            "--output_path",
            str(source_path / "distorted" / "sparse"),
        ]
    )
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)


def find_best_model_id(source_path: Path):
    models_dir = source_path / "distorted" / "sparse"
    models = list(sorted(models_dir.iterdir()))
    models = [x for x in models if x.is_dir()]
    assert len(models) > 0, f"No models found in {models_dir}"
    if len(models) == 1:
        return int(models[0].name)
    num_cameras = 0
    num_points = 0
    best_model = None
    for model in models:
        cameras = read_extrinsics_binary(model / "images.bin")
        if len(cameras) > num_cameras:
            num_cameras = len(cameras)
            best_model = model
        if len(cameras) == num_cameras or num_points == 0:
            points = read_points3D_binary(model / "points3D.bin")
            if len(points) > num_points:
                num_points = len(points)
                best_model = model
    assert best_model is not None
    return int(best_model.name)


def image_undistort(source_path: Path, colmap_command: str, model_id: int = 0):
    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = " ".join(
        [
            colmap_command,
            "image_undistorter",
            "--image_path",
            str(source_path / "input"),
            "--input_path",
            str(source_path / "distorted" / "sparse" / str(model_id)),
            "--output_path",
            str(source_path),
            "--output_type",
            "COLMAP",
        ]
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    (source_path / "sparse" / "0").mkdir(exist_ok=True, parents=True)
    # Copy each file from the source directory to the destination directory
    for file in (source_path / "sparse").iterdir():
        if file.name == "0" or file.is_dir():
            continue
        source_file = source_path / "sparse" / file.name
        destination_file = source_path / "sparse" / "0" / file.name
        shutil.move(source_file, destination_file)


def convert_model_to_text(source_path: Path, colmap_command: str):
    logging.info("Converting model to txt...")

    # Convert the model to txt.
    model_txt_cmd = " ".join(
        [
            colmap_command,
            "model_converter",
            "--input_path",
            str(source_path / "sparse" / "0"),
            "--output_path",
            str(source_path / "sparse" / "0"),
            "--output_type",
            "TXT",
        ]
    )
    exit_code = os.system(model_txt_cmd)
    if exit_code != 0:
        logging.error(f"Model conversion failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # remove the binary model with .bin extension
    for file in (source_path / "sparse" / "0").iterdir():
        if file.is_file() and file.suffix == ".bin":
            file.unlink()


def image_resize(source_path: Path, magick_command: str):
    logging.info("Copying and resizing...")

    # Resize images.
    for folder in ["images_2", "images_4", "images_8"]:
        (source_path / folder).mkdir(exist_ok=True)

    # Get the list of files in the source directory
    # Copy each file from the source directory to the destination directory
    for file in (source_path / "images").iterdir():
        source_file = source_path / "images" / file.name

        for folder, scale in [
            ("images_2", 0.5),
            ("images_4", 0.25),
            ("images_8", 0.125),
        ]:
            destination_file = source_path / folder / file.name
            shutil.copy2(source_file, destination_file)
            cmd = " ".join(
                [
                    magick_command,
                    "mogrify",
                    "-resize",
                    str(scale * 100) + "%",
                    str(destination_file),
                ]
            )
            exit_code = os.system(cmd)
            if exit_code != 0:
                logging.error(
                    f"{scale * 100}% resize failed with code {exit_code}. Exiting."
                )
                exit(exit_code)


if __name__ == "__main__":
    import argparse

    # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
    parser = argparse.ArgumentParser("SfM reconstruction with colmap/glomap")
    parser.add_argument("--source-path", "-s", required=True, type=str)
    parser.add_argument("--sfm", default="colmap", choices=["colmap", "glomap"])
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-matching", action="store_true")
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--colmap-executable", default="", type=str)
    parser.add_argument("--glomap-executable", default="", type=str)
    parser.add_argument("--magick-executable", default="", type=str)
    parser.add_argument(
        "--txt", action="store_true", help="Convert binary model to txt."
    )
    args = parser.parse_args()
    colmap_command = (
        '"{}"'.format(args.colmap_executable)
        if len(args.colmap_executable) > 0
        else "colmap"
    )
    glomap_command = (
        '"{}"'.format(args.glomap_executable)
        if len(args.glomap_executable) > 0
        else "glomap"
    )
    magick_command = (
        '"{}"'.format(args.magick_executable)
        if len(args.magick_executable) > 0
        else "magick"
    )
    use_gpu = 1 if not args.no_gpu else 0

    logging.info(f"Using {args.sfm}.")

    source_path = Path(args.source_path)

    if not (source_path / "input").exists():
        logging.error(
            "Input images not found. Please provide a valid source path with 'input' subdirectory."
        )
        exit()

    if not args.skip_matching:
        database_path = source_path / "distorted" / "database.db"
        feature_pipeline(
            source_path, database_path, colmap_command, args.camera, use_gpu
        )

        if args.sfm == "glomap":
            glomap(source_path, database_path, glomap_command)
        else:
            colmap(source_path, database_path, colmap_command)

    model_id = find_best_model_id(source_path)
    logging.info(f"Selecting best model with id '{model_id}'")
    image_undistort(source_path, colmap_command, model_id)

    if args.txt:
        convert_model_to_text(source_path, colmap_command)

    if args.resize:
        image_resize(source_path, magick_command)

    logging.info("Done.")
