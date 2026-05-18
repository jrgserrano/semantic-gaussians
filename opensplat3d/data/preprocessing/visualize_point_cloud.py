import argparse
import json
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as tf
from plyfile import PlyData


def load_ply(path: Path):
    """Load PLY file and return points and colors."""
    plydata = PlyData.read(str(path))
    vertices = plydata["vertex"]
    points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    return points, colors


def load_camera_json(json_path: Path):
    with json_path.open("r") as f:
        cameras = json.load(f)
    return cameras


def load_transforms_json(json_path: Path):
    with json_path.open("r") as f:
        meta = json.load(f)
    cameras = []
    for frame in meta.get("frames", []):
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        cameras.append({
            "position": c2w[:3, 3].tolist(),
            "rotation": c2w[:3, :3].tolist(),
            "width": meta.get("w", 640),
            "height": meta.get("h", 480),
            "fx": meta.get("fl_x", 525.0),
            "fy": meta.get("fl_y", meta.get("fl_x", 525.0)),
            "name": Path(frame.get("file_path", "")).stem,
        })
    return cameras


def add_camera_frustum(server: viser.ViserServer, camera_data, color=(255, 128, 64)):
    """Add camera frustum to viser server."""
    position = np.array(camera_data["position"])
    rotation = np.array(camera_data["rotation"])
    width = float(camera_data.get("width", 640))
    height = float(camera_data.get("height", 480))
    fx = float(camera_data.get("fx", 525.0))
    fy = float(camera_data.get("fy", fx))

    aspect = width / height
    fov_y = np.degrees(2.0 * np.arctan(height / (2.0 * fy)))
    wxyz = tf.SO3.from_matrix(rotation).wxyz

    server.scene.add_camera_frustum(
        name=f"camera/{camera_data.get('name', 'cam')}",
        fov=float(fov_y),
        aspect=float(aspect),
        scale=0.05,
        color=color,
        wxyz=wxyz,
        position=position,
        visible=True,
        variant="wireframe",
    )


def add_camera_centers(server: viser.ViserServer, cameras, color=(1.0, 0.0, 0.0), point_size=0.02):
    positions = np.array([camera["position"] for camera in cameras], dtype=np.float32)
    colors = np.tile(np.array(color, dtype=np.float32), (len(positions), 1))
    server.scene.add_point_cloud(
        name="camera_centers",
        points=positions,
        colors=colors,
        point_size=point_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a PLY point cloud and optional camera poses using viser (web-based)."
    )
    parser.add_argument("ply_path", type=Path, help="Path to the point cloud PLY file.")
    parser.add_argument(
        "--camera_json",
        type=Path,
        default=None,
        help="Optional camera JSON file (e.g. cameras.json) to render camera frustums.",
    )
    parser.add_argument(
        "--transforms_json",
        type=Path,
        default=None,
        help="Optional transforms.json file to load camera poses from the dataset output.",
    )
    parser.add_argument(
        "--show_axes",
        action="store_true",
        help="Show a world coordinate frame at the origin.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser web server.",
    )
    args = parser.parse_args()

    if args.camera_json is not None and args.transforms_json is not None:
        raise ValueError("Specify only one of --camera_json or --transforms_json.")

    if not args.ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {args.ply_path}")

    print(f"Loading point cloud from {args.ply_path}...")
    points, colors = load_ply(args.ply_path)
    print(f"Loaded {len(points)} points.")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")

    # Add point cloud
    server.scene.add_point_cloud(
        name="point_cloud",
        points=points,
        colors=colors,
        point_size=0.005,
    )

    cameras = []
    if args.camera_json is not None:
        if not args.camera_json.exists():
            raise FileNotFoundError(f"Camera JSON not found: {args.camera_json}")
        cameras = load_camera_json(args.camera_json)
    elif args.transforms_json is not None:
        if not args.transforms_json.exists():
            raise FileNotFoundError(f"Transforms JSON not found: {args.transforms_json}")
        cameras = load_transforms_json(args.transforms_json)

    if len(cameras) > 0:
        print(f"Loading {len(cameras)} cameras...")
        for camera in cameras:
            add_camera_frustum(server, camera)
        add_camera_centers(server, cameras)

    if args.show_axes:
        # Add coordinate frame
        server.scene.add_frame(
            "/world",
            wxyz=tf.SO3.from_x_radians(0.0).wxyz,
            position=(0, 0, 0),
            axes_length=0.2,
            axes_radius=0.01,
        )

    print(f"Visualization ready at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    server.sleep_forever()


if __name__ == "__main__":
    main()
