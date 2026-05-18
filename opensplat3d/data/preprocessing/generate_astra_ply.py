"""
Genera un archivo points3d.ply a partir del transforms.json y las profundidades
de la Astra, de forma similar a como se hace para el dataset Replica.

Uso:
    uv run python generate_astra_ply.py \
        --data /home/ubuntu/datasets/mapir_lab \
        --out /home/ubuntu/datasets/mapir_lab/points3d.ply \
        --blur 80.0 \
        --sample 0.02 \
        --max_depth 5.0
"""
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def laplacian_blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pose_is_far_enough(c2w, last_c2w, min_dist=0.1, min_angle_deg=5.0):
    """Devuelve True si la cámara se ha movido suficiente respecto al último frame."""
    if last_c2w is None:
        return True
    pos_cur = c2w[:3, 3]
    pos_last = last_c2w[:3, 3]
    dist = np.linalg.norm(pos_cur - pos_last)
    
    # Ángulo entre las direcciones de la cámara (columna Z de la rotación)
    dir_cur = c2w[:3, 2]
    dir_last = last_c2w[:3, 2]
    dot = np.clip(np.dot(dir_cur, dir_last), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))
    
    return dist >= min_dist or angle_deg >= min_angle_deg


def save_ply(points, colors, path):
    """Guarda una nube de puntos en formato PLY binario (compatible con create_from_pcd)."""
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(len(points), dtype=dtype)
    elements["x"] = points[:, 0]
    elements["y"] = points[:, 1]
    elements["z"] = points[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"] = colors[:, 0]
    elements["green"] = colors[:, 1]
    elements["blue"] = colors[:, 2]
    el = PlyElement.describe(elements, "vertex")
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(path))
    print(f"[OK] PLY guardado en {path} con {len(points):,} puntos.")


def generate_ply(data_dir, out_path, blur_threshold=80.0, sample_rate=0.02,
                 min_depth=0.3, max_depth=5.0, min_dist=0.1, min_angle_deg=5.0):
    data_dir = Path(data_dir)
    out_path = Path(out_path)

    json_path = data_dir / "transforms.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No se encuentra transforms.json en {data_dir}")

    with open(json_path, "r") as f:
        meta = json.load(f)

    fx = meta.get("fl_x", 535.4)
    fy = meta.get("fl_y", fx)
    cx = meta.get("cx", meta.get("w", 640) / 2)
    cy = meta.get("cy", meta.get("h", 480) / 2)
    W = meta.get("w", 640)
    H = meta.get("h", 480)

    frames = meta["frames"]
    print(f"[INFO] Procesando {len(frames)} frames...")
    print(f"       fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"       Filtro blur >= {blur_threshold}, muestra = {sample_rate*100:.1f}%")
    print(f"       Filtro pose: dist >= {min_dist}m o angulo >= {min_angle_deg}deg")

    all_points = []
    all_colors = []
    skipped_blur = 0
    skipped_depth = 0
    skipped_pose = 0
    last_c2w = None

    depth_dir = data_dir / "depth"

    for frame in tqdm(frames):
        # --- Rutas ---
        img_path = data_dir / frame["file_path"]
        if not img_path.exists():
            img_path = data_dir / "images" / Path(frame["file_path"]).name

        stem = Path(frame["file_path"]).stem
        depth_path = depth_dir / f"{stem}.png"

        # Buscar depth_file_path explícito si existe
        if "depth_file_path" in frame:
            depth_path = data_dir / frame["depth_file_path"]

        if not img_path.exists() or not depth_path.exists():
            skipped_depth += 1
            continue

        c2w = np.array(frame["transform_matrix"])

        # Filtro de redundancia por pose
        if not pose_is_far_enough(c2w, last_c2w, min_dist, min_angle_deg):
            skipped_pose += 1
            continue

        # --- Imagen y filtro de blur ---
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_depth += 1
            continue

        score = laplacian_blur_score(img)
        if score < blur_threshold:
            skipped_blur += 1
            continue

        # --- Profundidad ---
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth is None or depth.max() == 0:
            skipped_depth += 1
            continue
        depth = depth / 1000.0  # mm → metros

        # --- Máscara de validez ---
        mask = (depth > min_depth) & (depth < max_depth)
        # Submuestreo aleatorio para no generar una PCD de varios GB
        mask = mask & (np.random.rand(H, W) < sample_rate)

        if mask.sum() == 0:
            continue

        # --- Proyección 3D ---
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth[mask]
        x = (u[mask] - cx) * z / fx
        y = (v[mask] - cy) * z / fy

        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (N, 4)

        # Transformación Camera → World
        c2w = np.array(frame["transform_matrix"])
        pts_world = (c2w @ pts_cam.T).T[:, :3]  # (N, 3)
        last_c2w = c2w

        # --- Colores (BGR → RGB) ---
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols = img_rgb[v[mask], u[mask]]  # (N, 3)

        all_points.append(pts_world.astype(np.float32))
        all_colors.append(cols)

    print(f"\n[INFO] Descartados por blur: {skipped_blur} | por depth: {skipped_depth} | por pose: {skipped_pose}")

    if not all_points:
        print("[ERROR] No se generaron puntos. Revisa las rutas y filtros.")
        return

    pts = np.concatenate(all_points, axis=0)
    cols = np.concatenate(all_colors, axis=0)

    # Eliminar outliers extremos (percentil 1-99 por eje)
    for axis in range(3):
        lo, hi = np.percentile(pts[:, axis], [1, 99])
        valid = (pts[:, axis] >= lo) & (pts[:, axis] <= hi)
        pts = pts[valid]
        cols = cols[valid]

    print(f"[INFO] Puntos totales después de filtrado: {len(pts):,}")
    save_ply(pts, cols, out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Carpeta del dataset (con transforms.json, images/, depth/)")
    parser.add_argument("--out", default=None, help="Ruta de salida del PLY (default: data/points3d.ply)")
    parser.add_argument("--blur", type=float, default=80.0, help="Umbral de nitidez (Laplaciano)")
    parser.add_argument("--sample", type=float, default=0.02, help="Fracción de píxeles a muestrear")
    parser.add_argument("--min_depth", type=float, default=0.3, help="Profundidad mínima en metros")
    parser.add_argument("--max_depth", type=float, default=5.0, help="Profundidad máxima en metros")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Distancia mínima entre frames (metros)")
    parser.add_argument("--min_angle", type=float, default=5.0, help="Ángulo mínimo entre frames (grados)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(args.data) / "points3d.ply"
    generate_ply(args.data, out, args.blur, args.sample, args.min_depth, args.max_depth,
                 args.min_dist, args.min_angle)
