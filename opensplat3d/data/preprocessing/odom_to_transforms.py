import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def odom_to_nerfstudio(csv_path, output_dir, fx=535.4, fy=539.2, cx=320.1, cy=247.6):
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    
    # Extrínsecos EXACTOS de process_astra_colmap.py
    ext_pos = np.array([0.375, 0.0, 1.005])

    out = {
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "w": 640, "h": 480,
        "camera_model": "OPENCV",
        "is_astra": True,
        "frames": []
    }
    
    # Matrices de rotación de process_astra_colmap.py
    R_opt = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    T_cam_opt = np.eye(4)
    T_cam_opt[:3, :3] = R_opt

    # Matriz para pasar de OpenCV (Z-fwd) a Nerfstudio (Y-up, Z-back)
    flip_yz = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    print(f"[INFO] Convirtiendo {len(df)} poses usando LÓGICA COLMAP SCRIPT + NERFSTUDIO FLIP...")
    
    for _, row in df.iterrows():
        # 1. Pose del ROBOT (ROS)
        pos_robot = np.array([row['x'], row['y'], row['z']])
        quat_robot = [row['qx'], row['qy'], row['qz'], row['qw']]
        
        robot_pose = np.eye(4)
        robot_pose[:3, :3] = R.from_quat(quat_robot).as_matrix()
        robot_pose[:3, 3] = pos_robot
        
        # 2. T_robot_cam (Extrínsecos)
        T_robot_cam = np.eye(4)
        T_robot_cam[:3, 3] = ext_pos
        
        # 3. Combinación IGUAL que en el script de colmap
        c2w_opencv = robot_pose @ T_robot_cam @ T_cam_opt
        
        # 4. Ajuste para Nerfstudio (fundamental para que el Reader no vea las cámaras rotadas)
        final_transform = c2w_opencv @ flip_yz
        
        frame = {
            "file_path": f"images/{row['label']}",
            "transform_matrix": final_transform.tolist()
        }
        out["frames"].append(frame)
        
    with open(output_dir / "transforms.json", "w") as f:
        json.dump(out, f, indent=4)
    
    print(f"[OK] transforms.json generado. Esta configuración es idéntica a la que genera el PLY bien.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    odom_to_nerfstudio(args.csv, args.out)
