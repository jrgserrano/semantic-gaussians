import os
import cv2
import numpy as np
import torch
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
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} 0.0 0.0 0.0 {int(c[0])} {int(c[1])} {int(c[2])}\n")

def get_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_scannet(scannet_dir, output_dir, blur_thr=100.0, min_dist=0.05, min_angle=5.0, crop=25, denoise=True):
    scannet_dir = Path(scannet_dir)
    output_dir = Path(output_dir)
    
    # Crear estructura
    (output_dir / "color").mkdir(parents=True, exist_ok=True)
    (output_dir / "pose").mkdir(parents=True, exist_ok=True)
    (output_dir / "intrinsic").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    
    # Cargar y ajustar intrínsecos
    intrinsic_color_file = scannet_dir / "intrinsic" / "intrinsic_color.txt"
    intrinsic_depth_file = scannet_dir / "intrinsic" / "intrinsic_depth.txt"
    
    if not intrinsic_color_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de intrínsecos: {intrinsic_color_file}")

    K_color = np.loadtxt(intrinsic_color_file)
    K_depth = np.loadtxt(intrinsic_depth_file)
    
    # Obtener escalas entre color y profundidad
    color_files = sorted((scannet_dir / "color").glob("*.jpg"), key=lambda x: int(x.stem))
    temp_img = cv2.imread(str(color_files[0]))
    temp_depth = iio.imread(scannet_dir / "depth" / f"{color_files[0].stem}.png")
    
    scale_w = temp_depth.shape[1] / temp_img.shape[1]
    scale_h = temp_depth.shape[0] / temp_img.shape[0]
    
    # Ajustar centros de cámara por el recorte
    K_color[0, 2] -= crop
    K_color[1, 2] -= crop
    K_depth[0, 2] -= (crop * scale_w)
    K_depth[1, 2] -= (crop * scale_h)
    
    np.savetxt(output_dir / "intrinsic" / "intrinsic_color.txt", K_color)
    np.savetxt(output_dir / "intrinsic" / "intrinsic_depth.txt", K_depth)
    
    fx_d, fy_d, cx_d, cy_d = K_depth[0,0], K_depth[1,1], K_depth[0,2], K_depth[1,2]

    last_pose = None
    selected_indices = []
    all_points = []
    all_colors = []
    
    print(f"Procesando {len(color_files)} frames con crop={crop} y denoise={denoise}...")
    
    for img_path in tqdm(color_files):
        idx = img_path.stem
        pose_path = scannet_dir / "pose" / f"{idx}.txt"
        depth_path = scannet_dir / "depth" / f"{idx}.png"
        
        if not pose_path.exists() or not depth_path.exists():
            continue
            
        pose = np.loadtxt(pose_path)
        if np.isinf(pose).any(): continue
            
        img = cv2.imread(str(img_path))
        depth_raw = iio.imread(depth_path).astype(float)
        
        # 1. Filtro de Blur
        if get_blur_score(img) < blur_thr:
            continue
            
        # 2. Filtro de distancia/ángulo
        if last_pose is not None:
            dist = np.linalg.norm(pose[:3, 3] - last_pose[:3, 3])
            R_diff = np.dot(pose[:3, :3].T, last_pose[:3, :3])
            angle = np.rad2deg(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
            if dist < min_dist and angle < min_angle:
                continue
        
        # --- PREPROCESADO ---
        if crop > 0:
            img = img[crop:-crop, crop:-crop]
            cw, ch = int(crop * scale_w), int(crop * scale_h)
            depth_raw = depth_raw[ch:-ch, cw:-cw]
            
        if denoise:
            # Filtro bilateral para quitar ruido manteniendo bordes
            img = cv2.bilateralFilter(img, 9, 75, 75)
        # -------------------

        last_pose = pose
        selected_indices.append(idx)
        
        # Guardar procesados
        cv2.imwrite(str(output_dir / "color" / f"{idx}.jpg"), img)
        iio.imwrite(output_dir / "depth" / f"{idx}.png", depth_raw.astype(np.uint16))
        os.system(f"cp {pose_path} {output_dir / 'pose/'}")
        
        # 3. Proyectar puntos para PCD inicial
        depth = depth_raw / 1000.0
        h, w = depth.shape
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        
        mask = (depth > 0.5) & (depth < 5.0) 
        mask = mask & (np.random.rand(h, w) < 0.01)
        
        z = depth[mask]
        x = (uu[mask] - cx_d) * z / fx_d
        y = (vv[mask] - cy_d) * z / fy_d
        
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        pts_world = (pose @ pts_cam.T).T[:, :3]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color_u = (uu[mask] * img.shape[1] / w).astype(int)
        color_v = (vv[mask] * img.shape[0] / h).astype(int)
        cols = img_rgb[color_v, color_u]
        
        all_points.append(pts_world)
        all_colors.append(cols)

    # Guardar PCD final
    if all_points:
        pcd_pts = np.concatenate(all_points, axis=0)
        pcd_cols = np.concatenate(all_colors, axis=0)
        save_ply(pcd_pts, pcd_cols, output_dir / "points3d.ply")
        print(f"Frames seleccionados: {len(selected_indices)}")
        print(f"Nube de puntos guardada con {len(pcd_pts)} puntos.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blur", type=float, default=100.0)
    parser.add_argument("--dist", type=float, default=0.1) 
    parser.add_argument("--angle", type=float, default=10.0)
    parser.add_argument("--crop", type=int, default=25)
    parser.add_argument("--no_denoise", action="store_false", dest="denoise")
    args = parser.parse_args()
    process_scannet(args.input, args.output, args.blur, args.dist, args.angle, args.crop, args.denoise)
