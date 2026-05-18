"""
Convierte la salida de COLMAP/hloc (cameras.bin + images.bin) 
al formato transforms.json que necesita el AstraReader.

Uso:
    uv run python colmap_to_transforms.py \
        --colmap /home/ubuntu/datasets/mapir_lab/colmap/sparse \
        --out /home/ubuntu/datasets/mapir_lab \
        --images_prefix images
"""
import json
import struct
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            # Parámetros: OPENCV tiene 8 (fx, fy, cx, cy, k1, k2, p1, p2)
            # SIMPLE_RADIAL tiene 4, PINHOLE tiene 4...
            num_params = {0: 3, 1: 4, 2: 4, 3: 4, 4: 5, 5: 8, 6: 8, 7: 12}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {"w": w, "h": h, "params": params, "model_id": model_id}
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            # Saltar los puntos 2D
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_pts)  # x, y, point3D_id
            images[img_id] = {
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "cam_id": cam_id,
                "name": name.decode("utf-8"),
            }
    return images


def colmap_to_transforms(colmap_dir, output_dir, images_prefix="images"):
    colmap_dir = Path(colmap_dir)
    output_dir = Path(output_dir)

    cameras = read_cameras_binary(colmap_dir / "cameras.bin")
    images = read_images_binary(colmap_dir / "images.bin")

    # Usamos la primera cámara como referencia de intrínsecos
    cam = list(cameras.values())[0]
    params = cam["params"]
    w, h = cam["w"], cam["h"]

    # Para OPENCV (model_id=4) params = [fx, fy, cx, cy, ...]
    # Para SIMPLE_PINHOLE (0): [f, cx, cy]
    # Para PINHOLE (1): [fx, fy, cx, cy]
    model_id = cam["model_id"]
    if model_id == 0:  # SIMPLE_PINHOLE
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:  # PINHOLE, OPENCV y similares
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]

    out = {
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "w": w, "h": h,
        "camera_model": "OPENCV",
        "frames": []
    }

    for img_data in images.values():
        # Quaternion de COLMAP: (qw, qx, qy, qz) con convención w2c
        q_colmap = [img_data["qx"], img_data["qy"], img_data["qz"], img_data["qw"]]
        t_colmap = np.array([img_data["tx"], img_data["ty"], img_data["tz"]])

        # COLMAP da la transformación World→Camera (w2c), necesitamos Camera→World (c2w)
        rot_w2c = R.from_quat(q_colmap).as_matrix()
        t_w2c = t_colmap

        # Invertir: c2w = [R^T | -R^T * t]
        rot_c2w = rot_w2c.T
        t_c2w = -rot_c2w @ t_w2c

        c2w = np.eye(4)
        c2w[:3, :3] = rot_c2w
        c2w[:3, 3] = t_c2w

        # Cambio de eje COLMAP (x-right, y-down, z-front) → Nerfstudio (x-right, y-up, z-back)
        flip = np.diag([1, -1, -1, 1])
        c2w = c2w @ flip

        frame = {
            "file_path": f"{images_prefix}/{img_data['name']}",
            "transform_matrix": c2w.tolist()
        }
        out["frames"].append(frame)

    out_path = output_dir / "transforms.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"[OK] {len(out['frames'])} frames escritos en {out_path}")
    print(f"     Intrínsecos: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap", required=True, help="Carpeta sparse de COLMAP (con cameras.bin, images.bin)")
    parser.add_argument("--out", required=True, help="Carpeta de salida para transforms.json")
    parser.add_argument("--images_prefix", default="images", help="Prefijo de la carpeta de imágenes")
    args = parser.parse_args()

    colmap_to_transforms(args.colmap, args.out, args.images_prefix)
