import os
import sys
import subprocess
import json
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import sqlite3
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.interpolate import Rbf
from scipy.ndimage import distance_transform_edt

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Forzamos GPU 0
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_ply(points, colors, path):
    """Guarda una nube de puntos en formato PLY."""
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

class DepthStabilizer:
    def __init__(self, alpha=0.2):
        self.s_smoothed = None
        self.t_smoothed = None
        self.alpha = alpha

    def update(self, s_new, t_new):
        if self.s_smoothed is None:
            self.s_smoothed, self.t_smoothed = s_new, t_new
        else:
            # Filtro de paso bajo para los parámetros
            self.s_smoothed = (1 - self.alpha) * self.s_smoothed + self.alpha * s_new
            self.t_smoothed = (1 - self.alpha) * self.t_smoothed + self.alpha * t_new
        
        return self.s_smoothed, self.t_smoothed

class OpenSplatFullPipeline:
    def __init__(
        self,
        bag_path,
        out_dir,
        nth=1,
        max_frames=-1,
        min_dist=0.0,
        blur_threshold=0.0,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        pose_topic="/odom",
        min_angle=0.0,
        disable_pose_filter=False,
        use_odometry_poses=False,
        dedup_voxel_size=0.02,
        ext_pos=[0.375, 0.0, 1.005],
    ):
        self.bag_path = Path(bag_path)
        self.out_dir = Path(out_dir)
        self.img_dir = self.out_dir / "images"
        self.depth_dir = self.out_dir / "depth"
        self.depth_ia_dir = self.out_dir / "depth_ia"
        self.colmap_dir = self.out_dir / "colmap"
        self.nth = nth
        self.max_frames = max_frames
        self.min_dist = min_dist
        self.blur_threshold = blur_threshold
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.pose_topic = pose_topic
        self.min_angle = min_angle
        self.disable_pose_filter = disable_pose_filter
        self.use_odometry_poses = use_odometry_poses
        self.dedup_voxel_size = dedup_voxel_size
        self.ext_pos = np.array(ext_pos)
        
        # Matrices fijas de transformación
        self.T_robot_cam = np.eye(4)
        self.T_robot_cam[:3, 3] = self.ext_pos
        
        # Rotación de Optical (Z-fwd, X-right, Y-down) a ROS (X-fwd, Y-left, Z-up)
        # Col 0: Optical X (Right) -> ROS -Y (Left is +Y)
        # Col 1: Optical Y (Down)  -> ROS -Z (Up is +Z)
        # Col 2: Optical Z (Fwd)   -> ROS X
        self.R_opt = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ])

        #self.R_opt = np.eye(3)
        self.T_cam_opt = np.eye(4)
        self.T_cam_opt[:3, :3] = self.R_opt
        
        self.pose_timestamps = np.empty((0,), dtype=np.float64)
        self.pose_positions = np.empty((0, 3), dtype=np.float64)
        self.pose_quats = np.empty((0, 4), dtype=np.float64)
        self.frame_poses = []

        self.current_rgb = None

        for d in [self.img_dir, self.depth_dir, self.depth_ia_dir, self.colmap_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print("[INFO] Inicializando Depth Anything V2 Small...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(self.device)
        self.stabilizer = DepthStabilizer()

    def run_command(self, cmd, cwd=None, use_xvfb=True):
        """Ejecuta comandos de sistema (COLMAP)."""
        if use_xvfb:
            cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] + cmd
        print(f"Running: {' '.join(cmd)}")
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"Error running command: {result.stderr}")
            return None
        return result.stdout

    def align_depth(self, rel_depth, metric_depth):
        """Alineación híbrida: devuelve tanto la IA escalada como la versión combinada con Astra."""
        mask = (metric_depth > 0.5)
        rel_depth_inv = 1.0 / (rel_depth + 1e-6)
        
        if mask.sum() < 100:
            # Fallback: si no hay puntos suficientes para alinear, devolvemos la IA sin escalar
            return rel_depth_inv, metric_depth
            
        # Ratio global simple entre Astra e IA
        ratio = np.median(metric_depth[mask]) / np.median(rel_depth_inv[mask])
        d_ia = rel_depth_inv * ratio
        
        # Limpieza de valores inválidos
        #d_ia = np.nan_to_num(d_ia, nan=0.0, posinf=65.0, neginf=0.0)
        
        return d_ia, metric_depth

    def _compute_c2w(self, pos, quat):
        """Calcula la matriz Camera-to-World a partir de pose de robot [x,y,z,w]."""
        from scipy.spatial.transform import Rotation
        # Convertir quaternión de ROS [x,y,z,w] directamente a rotación
        R = Rotation.from_quat(quat).as_matrix()
        robot_pose = np.eye(4)
        robot_pose[:3, :3] = R
        robot_pose[:3, 3] = pos
        return robot_pose @ self.T_robot_cam @ self.T_cam_opt

    def _read_pose_history(self, reader):
        pose_conn = next((c for c in reader.connections if c.topic == self.pose_topic), None)
        if pose_conn is None:
            return

        entries = []
        for _, _, raw in reader.messages(connections=[pose_conn]):
            msg = reader.deserialize(raw, pose_conn.msgtype)
            if not hasattr(msg, "pose"):
                continue
            pose_msg = msg.pose
            if hasattr(pose_msg, "pose"):
                pose_msg = pose_msg.pose
            if not hasattr(pose_msg, "position") or not hasattr(pose_msg, "orientation"):
                continue
            stamp = None
            if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
                stamp = msg.header.stamp
            elif hasattr(pose_msg, "header") and hasattr(pose_msg.header, "stamp"):
                stamp = pose_msg.header.stamp
            if stamp is None:
                continue
            t = stamp.sec + stamp.nanosec * 1e-9
            pos = np.array([
                pose_msg.position.x,
                pose_msg.position.y,
                pose_msg.position.z,
            ], dtype=np.float64)
            quat = np.array([
                pose_msg.orientation.x,
                pose_msg.orientation.y,
                pose_msg.orientation.z,
                pose_msg.orientation.w,
            ], dtype=np.float64)
            entries.append((t, pos, quat))

        if not entries:
            print(f"[WARN] No pose entries found in topic {self.pose_topic}")
            return

        entries.sort(key=lambda x: x[0])
        self.pose_timestamps = np.array([x[0] for x in entries], dtype=np.float64)
        self.pose_positions = np.vstack([x[1] for x in entries])
        self.pose_quats = np.vstack([x[2] for x in entries])
        print(f"[INFO] Pose history: {len(entries)} entradas leídas")

    def _get_pose_at_time(self, timestamp):
        if self.pose_timestamps.size == 0:
            return None
        idx = np.searchsorted(self.pose_timestamps, timestamp)
        if idx == 0:
            return self.pose_positions[0], self.pose_quats[0]
        if idx >= len(self.pose_timestamps):
            return self.pose_positions[-1], self.pose_quats[-1]
        
        # Interpolación Lineal (LERP) para posición y Esférica (SLERP) para rotación
        t0, t1 = self.pose_timestamps[idx - 1], self.pose_timestamps[idx]
        alpha = (timestamp - t0) / (t1 - t0)
        
        pos = self.pose_positions[idx-1] + alpha * (self.pose_positions[idx] - self.pose_positions[idx-1])
        
        q0, q1 = self.pose_quats[idx-1], self.pose_quats[idx]
        # Implementación simple de SLERP para evitar dependencias extras si no hay scipy
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q0 = -q0
            dot = -dot
        if dot > 0.9995:
            quat = q0 + alpha * (q1 - q0)
            quat /= np.linalg.norm(quat)
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * alpha
            sin_theta = np.sin(theta)
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            quat = (s0 * q0) + (s1 * q1)
        
        return pos, quat

    def _quat_angle_diff(self, q0, q1):
        dot = np.dot(q0, q1)
        dot = np.clip(np.abs(dot), -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def _get_msg_timestamp(self, msg):
        stamp = None
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            stamp = msg.header.stamp
        elif hasattr(msg, "pose") and hasattr(msg.pose, "header") and hasattr(msg.pose.header, "stamp"):
            stamp = msg.pose.header.stamp
        if stamp is None:
            return None
        return stamp.sec + stamp.nanosec * 1e-9

    def _should_select_frame(self, current_pos, current_quat, last_pos, last_quat):
        if last_pos is None:
            return True
        dist = np.linalg.norm(current_pos - last_pos)
        if dist < self.min_dist:
            return False
        if self.min_angle > 0.0 and last_quat is not None:
            ang = self._quat_angle_diff(current_quat, last_quat)
            return ang >= self.min_angle
        return True

    def run_extraction(self):
        print(f"[1/4] Extrayendo Frames (Máx: {self.max_frames}, MinDist: {self.min_dist}m)...")
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        
        saved_count = 0
        last_pos = None
        last_quat = None

        with AnyReader([self.bag_path], default_typestore=typestore) as reader:
            try:
                color_conn = [c for c in reader.connections if 'color' in c.topic][0]
                depth_conn = [c for c in reader.connections if 'depth' in c.topic][0]
            except IndexError:
                print("Error: No se encontraron los tópicos de color/profundidad.")
                sys.exit(1)

            # Preleer la historia de poses para filtrar por movimiento del robot.
            self._read_pose_history(reader)

            c_msgs = []
            for _, _, raw in reader.messages(connections=[color_conn]):
                c_msg = reader.deserialize(raw, color_conn.msgtype)
                ts = self._get_msg_timestamp(c_msg)
                if ts is None:
                    continue
                c_msgs.append((ts, raw, c_msg))

            d_msgs = []
            for _, _, raw in reader.messages(connections=[depth_conn]):
                d_msg = reader.deserialize(raw, depth_conn.msgtype)
                ts = self._get_msg_timestamp(d_msg)
                if ts is None:
                    continue
                d_msgs.append((ts, raw, d_msg))

            depth_idx = 0
            for i in tqdm(range(0, len(c_msgs), self.nth), desc="Procesando"):
                if self.max_frames > 0 and saved_count >= self.max_frames:
                    break

                c_ts, _, c_msg = c_msgs[i]
                if depth_idx >= len(d_msgs):
                    break

                # Find the depth message closest to this color timestamp
                while depth_idx + 1 < len(d_msgs) and abs(d_msgs[depth_idx + 1][0] - c_ts) < abs(d_msgs[depth_idx][0] - c_ts):
                    depth_idx += 1

                d_ts, d_raw, d_msg = d_msgs[depth_idx]
                if abs(d_ts - c_ts) > 0.05:
                    # Skip if the closest depth frame is too far from the color frame
                    continue

                # Procesamiento
                c_msg = c_msg
                
                current_pos = None
                current_quat = None
                if self.pose_timestamps.size > 0 and hasattr(c_msg, "header") and hasattr(c_msg.header, "stamp"):
                    ts = c_msg.header.stamp.sec + c_msg.header.stamp.nanosec * 1e-9
                    pose = self._get_pose_at_time(ts)
                    if pose is not None:
                        current_pos, current_quat = pose

                if current_pos is None and hasattr(c_msg, "header") and hasattr(c_msg, "pose"):
                    current_pos = np.array([c_msg.pose.position.x, c_msg.pose.position.y, c_msg.pose.position.z])
                    current_quat = np.array([
                        c_msg.pose.orientation.x,
                        c_msg.pose.orientation.y,
                        c_msg.pose.orientation.z,
                        c_msg.pose.orientation.w,
                    ], dtype=np.float64)

                if self.use_odometry_poses and current_pos is None:
                    continue

                if not self.disable_pose_filter and current_pos is not None and not self._should_select_frame(current_pos, current_quat, last_pos, None if last_pos is None else last_quat):
                    continue

                # Procesamiento
                if "CompressedImage" in color_conn.msgtype:
                    img = cv2.imdecode(np.frombuffer(c_msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
                else:
                    img = np.frombuffer(c_msg.data, np.uint8).reshape(c_msg.height, c_msg.width, 3)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                self.current_rgb = img.copy()
                
                # --- FILTRADO DE DESENFOQUE ---
                if self.blur_threshold > 0:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if blur_score < self.blur_threshold:
                        # Opcional: print(f"Saltando frame borroso (score: {blur_score:.2f})")
                        continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                d_msg = reader.deserialize(d_raw, depth_conn.msgtype)
                d_astra = np.frombuffer(d_msg.data, np.uint16).reshape(d_msg.height, d_msg.width).astype(np.float32) / 1000.0
                
                inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    pred = self.model(**inputs).predicted_depth
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img.shape[:2], mode="bicubic").squeeze().cpu().numpy()
                
                # Alineamos la IA para obtener la escala correcta respecto a la Astra
                d_ia, d_final = self.align_depth(pred, d_astra)
                
                name = f"frame_{saved_count:05d}"
                # Guardamos la imagen RGB
                cv2.imwrite(str(self.img_dir / f"{name}.jpg"), img)
                
                # Filtramos la Astra (mediana 3x3) para limpiar ruido sin completar
                d_astra_filtered = cv2.medianBlur(d_astra.astype(np.float32), 3)
                
                # Guardamos Astra real filtrada en 'depth'
                cv2.imwrite(str(self.depth_dir / f"{name}.png"), (np.nan_to_num(d_astra_filtered, nan=0.0) * 1000).astype(np.uint16))
                # IA alineada para normales suaves (con clip de seguridad para evitar overflow)
                d_ia_uint = np.clip(np.nan_to_num(d_ia, nan=0.0) * 1000, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(self.depth_ia_dir / f"{name}.png"), d_ia_uint)
                
                if current_pos is not None and current_quat is not None:
                    self.frame_poses.append({
                        "name": name,
                        "position": current_pos.tolist(),
                        "quat": current_quat.tolist(),
                    })
                    last_pos = current_pos
                    last_quat = current_quat
                saved_count += 1

        print(f"[INFO] Frames extraídos y guardados: {saved_count}")
        return saved_count

    def inject_priors_to_colmap(self, db_path):
        """Inyecta las poses de odometría en la base de datos de COLMAP como priores."""
        print("[INFO] Inyectando poses de odometría en la base de datos de COLMAP...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Mapear nombres de imagen a IDs
        cursor.execute("SELECT image_id, name FROM images")
        db_images = {name: img_id for img_id, name in cursor.fetchall()}
        
        for frame in self.frame_poses:
            name = f"{frame['name']}.jpg"
            if name not in db_images: continue
            
            img_id = db_images[name]
            c2w = self._compute_c2w(np.array(frame["position"]), np.array(frame["quat"]))
            w2c = np.linalg.inv(c2w)
            
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            
            # Convertir R a quat [w,x,y,z] para COLMAP
            tr = np.trace(R)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2
                qw, qx, qy, qz = 0.25 * S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S
            else:
                if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
                    S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                    qw, qx, qy, qz = (R[2,1]-R[1,2])/S, 0.25 * S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S
                elif R[1,1] > R[2,2]:
                    S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                    qw, qx, qy, qz = (R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25 * S, (R[1,2]+R[2,1])/S
                else:
                    S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                    qw, qx, qy, qz = (R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25 * S

            cursor.execute("""
                UPDATE images 
                SET prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=?, 
                    prior_tx=?, prior_ty=?, prior_tz=? 
                WHERE image_id=?
            """, (qw, qx, qy, qz, T[0], T[1], T[2], img_id))
            
        conn.commit()
        conn.close()

    def run_colmap(self):
        print("[2/4] Ejecutando COLMAP refinado con Odometría...")
        db = self.colmap_dir / "database.db"
        sparse = self.colmap_dir / "sparse"
        sparse.mkdir(exist_ok=True)

        # 1. Extracción de características
        cmd_extract = ["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(self.img_dir), "--ImageReader.single_camera", "1"]
        if self.fx is not None:
            # Usamos OPENCV en lugar de PINHOLE para permitir que COLMAP modele la distorsión
            # Parámetros: fx, fy, cx, cy, k1, k2, p1, p2
            cmd_extract += ["--ImageReader.camera_model", "OPENCV", "--ImageReader.camera_params", f"{self.fx},{self.fy},{self.cx},{self.cy},0,0,0,0"]
        else:
            cmd_extract += ["--ImageReader.camera_model", "OPENCV"]
        self.run_command(cmd_extract)

        # 2. Inyectar Priores si hay poses disponibles
        if self.frame_poses:
            self.inject_priors_to_colmap(db)

        # 3. Matching
        self.run_command(["colmap", "exhaustive_matcher", "--database_path", str(db)])

        # 4. Mapper con Priores
        cmd_map = [
            "colmap", "mapper", 
            "--database_path", str(db), 
            "--image_path", str(self.img_dir), 
            "--output_path", str(sparse),
            "--Mapper.use_priors", "1",
            "--Mapper.ba_use_priors", "1"
        ]
        self.run_command(cmd_map)
        
        text_dir = self.colmap_dir / "text"
        text_dir.mkdir(exist_ok=True)
        if (sparse / "0").exists():
            self.run_command(["colmap", "model_converter", "--input_path", str(sparse / "0"), "--output_path", str(text_dir), "--output_type", "TXT"])
            return text_dir
        else:
            print("[WARN] COLMAP con priores falló. Intentando sin priores...")
            self.run_command(["colmap", "mapper", "--database_path", str(db), "--image_path", str(self.img_dir), "--output_path", str(sparse)])
            if (sparse / "0").exists():
                self.run_command(["colmap", "model_converter", "--input_path", str(sparse / "0"), "--output_path", str(text_dir), "--output_type", "TXT"])
                return text_dir
            return None

    def create_dataset(self, colmap_text=None, target_points=1000000):
        print("[3/4] Generando transforms.json y PCD Densa...")
        
        frames, pcd_pts, pcd_cols = [], [], []

        if self.use_odometry_poses:
            if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
                raise ValueError("Se requieren intrínsecos de cámara para usar poses de odometría.")

            sample_depth = next(self.depth_dir.glob("*.png"), None)
            if sample_depth is None:
                raise FileNotFoundError("No se encontró ninguna profundidad en el directorio de salida.")
            sample_img = cv2.imread(str(sample_depth), cv2.IMREAD_UNCHANGED)
            h, w = sample_img.shape[:2]
            fl_x, fl_y, cx, cy = self.fx, self.fy, self.cx, self.cy

            seen_voxels = set()
            for frame in self.frame_poses:
                name = f"{frame['name']}.jpg"
                c2w = self._compute_c2w(np.array(frame["position"]), np.array(frame["quat"]))

                depth_path = self.depth_dir / f"{frame['name']}.png"
                frames.append({
                    "file_path": f"images/{name}",
                    "depth_file_path": f"depth/{frame['name']}.png",
                    "transform_matrix": c2w.tolist(),
                })

                if depth_path.exists():
                    d_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    c_img = cv2.cvtColor(cv2.imread(str(self.img_dir / name)), cv2.COLOR_BGR2RGB)
                    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
                    z = d_img.astype(float) / 1000.0
                    mask = (z > 0.5) & (z < 8.0)
                    if not np.any(mask):
                        continue

                    pts = (c2w @ np.stack([
                        (uu[mask] - cx) * z[mask] / fl_x,
                        (vv[mask] - cy) * z[mask] / fl_y,
                        z[mask],
                        np.ones_like(z[mask])
                    ], axis=-1).T).T[:, :3]

                    if len(pts) > 0:
                        voxel_idx = np.floor(pts / self.dedup_voxel_size).astype(np.int64)
                        keep_mask = []
                        for j, vox in enumerate(map(tuple, voxel_idx)):
                            if vox in seen_voxels:
                                keep_mask.append(False)
                            else:
                                seen_voxels.add(vox)
                                keep_mask.append(True)
                        keep_mask = np.array(keep_mask, dtype=bool)
                        pts = pts[keep_mask]
                        cols = c_img[mask][keep_mask]
                    else:
                        cols = np.empty((0, 3), dtype=np.uint8)

                    if len(pts) > 0:
                        n_pts = min(len(pts), int(target_points / max(len(self.frame_poses), 1) * 1.5))
                        if len(pts) > n_pts:
                            idx = np.random.choice(len(pts), n_pts, replace=False)
                            pts = pts[idx]
                            cols = cols[idx]
                        pcd_pts.extend(pts.tolist())
                        pcd_cols.extend(cols.tolist())
        else:
            if colmap_text is None:
                raise ValueError("Se requiere el texto COLMAP cuando no se usan poses de odometría.")

            with open(colmap_text / "cameras.txt", "r") as f:
                for line in f:
                    if line.startswith("#"): continue
                    elems = line.split()
                    fl_x, fl_y, cx, cy = float(elems[4]), float(elems[5]), float(elems[6]), float(elems[7])
                    w, h = int(elems[2]), int(elems[3])
                    break

            with open(colmap_text / "images.txt", "r") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                for i in range(0, len(lines), 2):
                    elems = lines[i].split()
                    q = np.array([float(elems[1]), float(elems[2]), float(elems[3]), float(elems[4])])
                    t = np.array([float(elems[5]), float(elems[6]), float(elems[7])])
                    name = elems[9]

                    R = np.array([
                        [1-2*q[2]**2-2*q[3]**2, 2*q[1]*q[2]-2*q[0]*q[3], 2*q[3]*q[1]+2*q[0]*q[2]],
                        [2*q[1]*q[2]+2*q[0]*q[3], 1-2*q[1]**2-2*q[3]**2, 2*q[2]*q[3]-2*q[0]*q[1]],
                        [2*q[3]*q[1]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], 1-2*q[1]**2-2*q[2]**2]
                    ])
                    c2w = np.eye(4)
                    c2w[:3, :3] = R
                    c2w[:3, 3] = t

                    depth_path = self.depth_dir / name.replace(".jpg", ".png")
                    frames.append({
                        "file_path": f"images/{name}",
                        "depth_file_path": f"depth/{name.replace('.jpg', '.png')}",
                        "transform_matrix": c2w.tolist(),
                    })

                    if depth_path.exists():
                        d_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        c_img = cv2.cvtColor(cv2.imread(str(self.img_dir / name)), cv2.COLOR_BGR2RGB)
                        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
                        z = d_img.astype(float) / 1000.0
                        mask = (z > 0.5) & (z < 10.0)
                        if not np.any(mask):
                            continue
                        pts = (c2w @ np.stack([
                            (uu[mask] - cx) * z[mask] / fl_x,
                            (vv[mask] - cy) * z[mask] / fl_y,
                            z[mask],
                            np.ones_like(z[mask])
                        ], axis=-1).T).T[:, :3]
                        n_pts = min(len(pts), int(target_points / (len(lines)/2) * 1.5))
                        idx = np.random.choice(len(pts), n_pts, replace=False)
                        pcd_pts.extend(pts[idx].tolist())
                        pcd_cols.extend(c_img[mask][idx].tolist())

        # Guardar transforms.json compatible con AstraReader
        with open(self.out_dir / "transforms.json", "w") as f:
            json.dump({
                "is_astra": True, 
                "fl_x": fl_x, "fl_y": fl_y, 
                "cx": cx, "cy": cy, 
                "w": w, "h": h, 
                "frames": frames
            }, f, indent=4)
            
        print(f"Guardando {len(pcd_pts)} puntos en points3d.ply...")
        save_ply(pcd_pts, pcd_cols, self.out_dir / "points3d.ply")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Ruta al bag de ROS2")
    parser.add_argument("--out", default="./astra_colmap_dataset", help="Directorio de salida")
    parser.add_argument("--nth", type=int, default=1, help="Frecuencia de frames")
    parser.add_argument("--max_frames", type=int, default=-1, help="Máximo de frames a extraer")
    parser.add_argument("--min_dist", type=float, default=0.0, help="Distancia mínima entre frames (metros)")
    parser.add_argument("--blur_threshold", type=float, default=0.0, help="Umbral de nitidez (Laplaciano). Sugerido: 100")
    parser.add_argument("--fx", type=float, default=516.4535522460938, help="Focal X del sensor Astra")
    parser.add_argument("--fy", type=float, default=516.4535522460938, help="Focal Y del sensor Astra")
    parser.add_argument("--cx", type=float, default=332.4849548339844, help="Centro óptico X del sensor Astra")
    parser.add_argument("--cy", type=float, default=242.23336791992188, help="Centro óptico Y del sensor Astra")
    parser.add_argument("--pose_topic", type=str, default="/amcl_pose", help="Tópico ROS2 de odometría para filtrar frames")
    parser.add_argument("--min_angle", type=float, default=0.0, help="Ángulo mínimo entre frames consecutivos (grados)")
    parser.add_argument("--disable_pose_filter", action="store_true", help="Desactiva el filtrado de frames por pose (usa solo distancia/ángulo)")
    parser.add_argument("--use_odometry_poses", action="store_true", help="Construye transforms.json y la nube usando poses de odometría en lugar de COLMAP.")
    parser.add_argument("--dedup_voxel_size", type=float, default=0.02, help="Tamaño de voxel en metros para deduplicar puntos de superposición entre imágenes.")
    parser.add_argument("--ext_x", type=float, default=0.375, help="Offset X de la cámara respecto al robot")
    parser.add_argument("--ext_y", type=float, default=0.0, help="Offset Y de la cámara respecto al robot")
    parser.add_argument("--ext_z", type=float, default=1.005, help="Offset Z de la cámara respecto al robot")
    args = parser.parse_args()

    pipeline = OpenSplatFullPipeline(
        args.bag,
        args.out,
        args.nth,
        args.max_frames,
        args.min_dist,
        args.blur_threshold,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        pose_topic=args.pose_topic,
        min_angle=args.min_angle,
        disable_pose_filter=args.disable_pose_filter,
        use_odometry_poses=args.use_odometry_poses,
        dedup_voxel_size=args.dedup_voxel_size,
        ext_pos=[args.ext_x, args.ext_y, args.ext_z]
    )
    pipeline.run_extraction()
    txt_path = None if args.use_odometry_poses else pipeline.run_colmap()
    if args.use_odometry_poses or txt_path:
        pipeline.create_dataset(txt_path)
        print(f"\n¡Proceso completado! Dataset listo en: {args.out}")
    else:
        print("\nError: No se pudo completar la reconstrucción con COLMAP.")
