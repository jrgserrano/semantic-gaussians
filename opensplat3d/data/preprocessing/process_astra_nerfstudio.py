import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

class AstraProExtractor:
    def __init__(self, bag_path, out_path, image_topic="/astra_camera/color/image_raw", 
                 depth_topic="/astra_camera/depth/image_raw", odom_topic="/amcl_pose"):
        self.bag_path = Path(bag_path)
        self.out_path = Path(out_path)
        self.image_topic = image_topic
        self.depth_topic = depth_topic
        self.odom_topic = odom_topic
        
        # Estructura de salida
        self.img_dir = self.out_path / "images"
        self.depth_dir = self.out_path / "depth"
        for d in [self.img_dir, self.depth_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

    def apply_bilateral_filter(self, depth_mm):
        depth_m = depth_mm.astype(np.float32) / 1000.0
        filtered = cv2.bilateralFilter(depth_m, 5, 0.1, 5)
        return (filtered * 1000.0).astype(np.uint16)

    def blur_score(self, img_bgr):
        """Varianza del Laplaciano: cuanto mayor, más nítida la imagen."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def pose_is_far_enough(self, odom, last_odom, min_dist=0.1, min_angle_deg=5.0):
        """Devuelve True si la pose actual es suficientemente distinta de la anterior."""
        if last_odom is None:
            return True
        _, x0, y0, z0, qx0, qy0, qz0, qw0 = last_odom
        _, x1, y1, z1, qx1, qy1, qz1, qw1 = odom
        
        # Distancia traslacional
        dist = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        
        # Distancia rotacional: ángulo entre quaterniones
        dot = abs(qx0*qx1 + qy0*qy1 + qz0*qz1 + qw0*qw1)
        dot = np.clip(dot, 0.0, 1.0)
        angle_deg = 2 * np.degrees(np.arccos(dot))
        
        return dist >= min_dist or angle_deg >= min_angle_deg

    def extract(self, skip_frames=0, blur_threshold=80.0, min_dist=0.1, min_angle_deg=5.0):
        image_msgs = {}
        depth_msgs = {}
        odom_msgs = []  # (timestamp, x, y, z, qx, qy, qz, qw)
        
        print(f"[INFO] Leyendo Bag: {self.bag_path}")
        
        with AnyReader([self.bag_path], default_typestore=self.typestore) as reader:
            connections = [c for c in reader.connections if c.topic in [self.image_topic, self.depth_topic, self.odom_topic]]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = self.typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                if connection.topic == self.image_topic:
                    image_msgs[timestamp] = msg
                elif connection.topic == self.depth_topic:
                    depth_msgs[timestamp] = msg
                elif connection.topic == self.odom_topic:
                    msg_type = connection.msgtype
                    # PoseWithCovarianceStamped (/amcl_pose)
                    if "PoseWithCovarianceStamped" in msg_type:
                        p = msg.pose.pose.position
                        o = msg.pose.pose.orientation
                    # Odometry (/odom)
                    elif "Odometry" in msg_type:
                        p = msg.pose.pose.position
                        o = msg.pose.pose.orientation
                    # PoseStamped (fallback)
                    else:
                        p = msg.pose.position
                        o = msg.pose.orientation
                    odom_msgs.append((timestamp, p.x, p.y, p.z, o.x, o.y, o.z, o.w))

        print("[INFO] Sincronizando y guardando frames...")
        img_timestamps = sorted(image_msgs.keys())
        depth_timestamps = sorted(depth_msgs.keys())
        
        poses_file = open(self.out_path / "poses_odometry.csv", "w")
        poses_file.write("label,x,y,z,qx,qy,qz,qw\n")
        
        saved_count = 0
        last_saved_odom = None
        for i, t_img in enumerate(tqdm(img_timestamps)):
            if i % (skip_frames + 1) != 0:
                continue
                
            t_depth = min(depth_timestamps, key=lambda x: abs(x - t_img))
            if abs(t_depth - t_img) > 500_000_000:
                continue
            
            best_odom = min(odom_msgs, key=lambda x: abs(x[0] - t_img)) if odom_msgs else None

            # Filtro de redundancia por pose
            if best_odom and not self.pose_is_far_enough(best_odom, last_saved_odom, min_dist, min_angle_deg):
                continue

            img_msg = image_msgs[t_img]
            depth_msg = depth_msgs[t_depth]
            
            # Imagen RGB
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Filtro de nitidez: descartar imágenes borrosas
            score = self.blur_score(img_bgr)
            if score < blur_threshold:
                continue
            
            # Profundidad con Filtro Bilateral
            depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width)
            depth_filt = self.apply_bilateral_filter(depth)
            
            name = f"frame_{saved_count:05d}"
            cv2.imwrite(str(self.img_dir / f"{name}.jpg"), img_bgr)
            cv2.imwrite(str(self.depth_dir / f"{name}.png"), depth_filt)
            
            if best_odom:
                _, x, y, z, qx, qy, qz, qw = best_odom
                poses_file.write(f"{name}.jpg,{x},{y},{z},{qx},{qy},{qz},{qw}\n")
                last_saved_odom = best_odom
            
            saved_count += 1

        poses_file.close()
        print(f"\n[OK] {saved_count} frames guardados.")
        print(f"[INFO] CSV de poses: {self.out_path / 'poses_odometry.csv'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--odom", default="/amcl_pose")
    parser.add_argument("--blur", type=float, default=80.0, help="Umbral de nitidez (Laplaciano). Menor = menos restrictivo")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Distancia mínima entre frames (metros)")
    parser.add_argument("--min_angle", type=float, default=5.0, help="Ángulo mínimo entre frames (grados)")
    args = parser.parse_args()
    
    extractor = AstraProExtractor(args.bag, args.out, odom_topic=args.odom)
    extractor.extract(skip_frames=args.skip, blur_threshold=args.blur,
                      min_dist=args.min_dist, min_angle_deg=args.min_angle)
