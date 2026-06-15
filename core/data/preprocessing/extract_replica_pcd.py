import argparse
import numpy as np
import imageio.v3 as iio
from pathlib import Path

# core imports
from core.utils.scene_utils import store_ply

def parse_traj_txt(path):
    c2ws = []
    with open(path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                c2w = np.array(values, dtype=np.float32).reshape(4, 4)
                c2ws.append(c2w)
    return c2ws

def main(args):
    dataset_path = Path(args.dataset_path)
    images_dir = dataset_path / "results"
    traj_path = dataset_path / "traj.txt"
    out_ply = dataset_path / "points3d.ply"
    
    if not traj_path.exists():
        print(f"Error: {traj_path} no encontrado.")
        return

    # read the transformation matrices (Camera to World)
    c2ws = parse_traj_txt(traj_path)
    
    # Pinhole Camera Model
    width, height = 1200, 680
    fx, fy = 600.0, 600.0
    cx, cy = width / 2.0, height / 2.0
    
    all_xyz = []
    all_rgb = []
    
    # process sparse frames to capture the whole room
    step = max(1, len(c2ws) // args.num_frames)
    print(f"Processing {args.num_frames} frames from {dataset_path}...")
    
    # ray-casting
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    for idx in range(0, len(c2ws), step):
        c2w = c2ws[idx]
        
        img_path = images_dir / f"frame{idx:06d}.jpg"
        depth_path = images_dir / f"depth{idx:06d}.png"
        
        if not img_path.exists() or not depth_path.exists():
            continue
            
        # extract color and depth
        rgb = iio.imread(img_path)
        depth = iio.imread(depth_path).astype(np.float32)
        
        # convert raw scalar from dataset to floating meters
        depth = depth / 6553.5
        
        # mask to reject blind pixels (infinites or very close collisions)
        valid_mask = (depth > 0.1) & (depth < 8.0)
        
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        valid_z = depth[valid_mask]
        valid_rgb = rgb[valid_mask]
        
        cam_X = (valid_u - cx) * valid_z / fx
        cam_Y = (valid_v - cy) * valid_z / fy
        cam_Z = valid_z
        
        # homogenize points to be able to multiply them by 4x4 matrices
        cam_pts = np.stack([cam_X, cam_Y, cam_Z, np.ones_like(cam_Z)], axis=1)
        
        # transform to world space
        world_pts = (c2w @ cam_pts.T).T
        
        # record the coordinates ignoring the homogeneous '1'
        all_xyz.append(world_pts[:, :3])
        all_rgb.append(valid_rgb)
        
        print(f"  Frame {idx:04d} geometrically assembled ({len(valid_z)} points)")

    # concatenation and sub-sampling
    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    
    num_points = all_xyz.shape[0]
    
    if args.max_points > 0 and args.max_points < num_points:
        target_points = args.max_points
        # if we have too many points, we choose them randomly to not overflow the VRAM
        indices = np.random.choice(num_points, target_points, replace=False)
        final_xyz = all_xyz[indices]
        final_rgb = all_rgb[indices]
    else:
        target_points = num_points
        final_xyz = all_xyz
        final_rgb = all_rgb
    
    store_ply(out_ply, final_xyz, final_rgb)
    print(f"\nCloud intercepted and saved to: {out_ply}")
    print(f"Total base points that will inherit the Gaussians: {target_points}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ground Truth Depth to Initial 3D PLY Cloud")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset (ej: /home/ubuntu/datasets/Replica/room0)")
    parser.add_argument("--num_frames", type=int, default=15, help="Max number of sparse frames to process")
    parser.add_argument("--max_points", type=int, default=100000, help="Budget for initial PLY cloud")
    
    args = parser.parse_args()
    main(args)
