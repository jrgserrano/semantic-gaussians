import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio.v3 as iio

def save_ply(points, colors, path):
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

def get_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def generate_pcd(nerfstudio_path, blur_threshold=100.0, sample_rate=0.01):
    path = Path(nerfstudio_path)
    json_path = path / "transforms.json"
    
    if not json_path.exists():
        print(f"[ERROR] No se encuentra transforms.json en {path}")
        return

    with open(json_path, "r") as f:
        meta = json.load(f)

    # Obtener intrínsecos de Nerfstudio
    fx = meta.get("fl_x")
    fy = meta.get("fl_y")
    cx = meta.get("cx")
    cy = meta.get("cy")
    w_json = meta.get("w")
    h_json = meta.get("h")

    frames = meta["frames"]
    all_points = []
    all_colors = []
    
    print(f"[INFO] Generando PCD desde {len(frames)} frames optimizados...")
    
    for frame in tqdm(frames):
        img_path = path / frame["file_path"]
        # El nombre del frame suele ser frame_XXXXX.jpg, buscamos el .png equivalente en depth/
        depth_name = Path(img_path).stem + ".png"
        depth_path = path / "depth" / depth_name
        
        if not depth_path.exists():
            continue
            
        # 1. Filtro de Blur (opcional)
        img = cv2.imread(str(img_path))
        if get_blur_score(img) < blur_threshold:
            continue
            
        # 2. Cargar profundidad (ya filtrada con bilateral) y Pose
        depth = iio.imread(depth_path).astype(float) / 1000.0
        c2w = np.array(frame["transform_matrix"])
        
        # 3. Proyectar puntos
        h, w = depth.shape
        # Ajustar intrínsecos si la resolución de la imagen es distinta a la del JSON
        scale_x = w / w_json
        scale_y = h / h_json
        
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        
        # Máscara de validez: 0.5m a 5m para evitar ruido de fondo
        mask = (depth > 0.5) & (depth < 5.0)
        # Submuestreo para no crear una PCD de gigabytes
        mask = mask & (np.random.rand(h, w) < sample_rate)
        
        z = depth[mask]
        x = (uu[mask] - (cx * scale_x)) * z / (fx * scale_x)
        y = (vv[mask] - (cy * scale_y)) * z / (fy * scale_y)
        
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        pts_world = (c2w @ pts_cam.T).T[:, :3]
        
        # Colores
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols = img_rgb[vv[mask], uu[mask]]
        
        all_points.append(pts_world)
        all_colors.append(cols)

    if all_points:
        pcd_pts = np.concatenate(all_points, axis=0)
        pcd_cols = np.concatenate(all_colors, axis=0)
        output_ply = path / "points3d.ply"
        save_ply(pcd_pts, pcd_cols, output_ply)
        print(f"\n[OK] Nube de puntos guardada en {output_ply} con {len(pcd_pts)} puntos.")
    else:
        print("[ERROR] No se generaron puntos. Revisa los filtros o las rutas.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Carpeta del dataset de Nerfstudio")
    parser.add_argument("--blur", type=float, default=50.0, help="Umbral de nitidez")
    parser.add_argument("--sample", type=float, default=0.01, help="Tasa de submuestreo (0.01 = 1%)")
    args = parser.parse_args()
    
    generate_pcd(args.data, args.blur, args.sample)
