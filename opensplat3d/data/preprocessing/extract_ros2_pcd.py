
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# opensplat3d imports
from opensplat3d.data.ros2_reader import ROS2Reader
from opensplat3d.utils.scene_utils import store_ply

def main(args):
    bag_path = Path(args.bag_path)
    out_ply = bag_path.parent / f"{bag_path.stem}_points3d.ply"
    
    # Intrinsics for Astra Lab (from user)
    intrinsics = {
        "fx": 516.4535522460938,
        "fy": 516.4535522460938,
        "cx": 332.4849548339844,
        "cy": 242.23336791992188
    }
    
    # Re-enable depth loading temporarily for this script
    # Note: ROS2Reader has depth commented out in _process, we need to make sure it works.
    
    reader = ROS2Reader(
        str(bag_path),
        world_frame=args.world_frame,
        camera_frame=args.camera_frame,
        intrinsics=intrinsics,
        nth_frames=args.nth_frames,
        num_frames=args.max_frames if args.max_frames > 0 else -1,
    )
    
    print(f"[PCD] Extracting point cloud from {len(reader.frames)} frames...")
    
    all_xyz = []
    all_rgb = []
    
    # We need to manually handle depth loading since we commented it out in the main reader
    # To avoid modifying the main reader again, let's just do it here
    from rosbags.highlevel import AnyReader
    
    with AnyReader([bag_path], default_typestore=reader.typestore) as r:
        max_idx = min(len(reader.frames), args.max_frames) if args.max_frames > 0 else len(reader.frames)
        for i in tqdm(range(max_idx)):
            frame = reader.frames[i]
            
            # Get color and depth directly
            msg_color = r.deserialize(frame["color_raw"], frame["color_conn"].msgtype)
            img = np.frombuffer(msg_color.data, dtype=np.uint8).reshape(msg_color.height, msg_color.width, 3)
            
            msg_depth = r.deserialize(frame["depth_raw"], frame["depth_conn"].msgtype)
            depth = np.frombuffer(msg_depth.data, dtype=np.uint16).reshape(msg_depth.height, msg_depth.width).astype(np.float32) / 1000.0
            
            # Project to 3D
            h, w = depth.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            
            valid = (depth > 0.2) & (depth < 5.0) # Reject noise
            z = depth[valid]
            x = (u[valid] - intrinsics["cx"]) * z / intrinsics["fx"]
            y = (v[valid] - intrinsics["cy"]) * z / intrinsics["fy"]
            
            cam_pts = np.stack([x, y, z, np.ones_like(z)], axis=1)
            
            # Transform to world space
            # c2w is already calculated by the reader
            c2w = frame["pose"]
            world_pts = (c2w @ cam_pts.T).T[:, :3]
            
            all_xyz.append(world_pts)
            all_rgb.append(img[valid])
            
    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    
    # Subsample if too many points
    if args.max_points > 0 and len(all_xyz) > args.max_points:
        indices = np.random.choice(len(all_xyz), args.max_points, replace=False)
        all_xyz = all_xyz[indices]
        all_rgb = all_rgb[indices]
        
    store_ply(out_ply, all_xyz, all_rgb)
    print(f"\n[PCD] Initial cloud saved to: {out_ply}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", type=str)
    parser.add_argument("--nth_frames", type=int, default=1) # Use fewer frames for PCD
    parser.add_argument("--max_points", type=int, default=100000)
    parser.add_argument("--world_frame", type=str, default="map")
    parser.add_argument("--camera_frame", type=str, default="astra_camera_color_optical_frame")
    parser.add_argument("--max_frames", type=int, default=0, help="Max number of frames to process")
    args = parser.parse_args()
    main(args)
