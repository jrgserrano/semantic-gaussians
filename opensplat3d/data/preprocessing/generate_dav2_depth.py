
import argparse
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# opensplat3d imports
from opensplat3d.data.ros2_reader import ROS2Reader

def align_depth(source, target):
    """
    Finds scale (a) and shift (b) in the reciprocal domain (disparity).
    Most monocular models output values proportional to 1/Z.
    target_inv = a * source + b
    """
    mask = (target > 0.5) & (target < 5.0) # Use astra's reliable range
    if mask.sum() < 100:
        return source # Not enough data
    
    # We work in inverse depth (disparity)
    target_inv = 1.0 / target[mask]
    source_valid = source[mask]
    
    # Solve Ax = B for disparity: 1/Z_metric = a * D_ia + b
    A = np.vstack([source_valid, np.ones_like(source_valid)]).T
    a, b = np.linalg.lstsq(A, target_inv, rcond=None)[0]
    
    # Compute aligned disparity and convert back to metric depth
    aligned_inv = a * source + b
    aligned_inv = np.clip(aligned_inv, 1.0/10.0, 1.0/0.1) # Bound between 0.1m and 10m
    
    return 1.0 / aligned_inv

def main(args):
    bag_path = Path(args.bag_path)
    output_dir = bag_path.parent / "depth_dav2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DAV2] Initializing Depth Anything v2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We use Depth-Anything-V2-Small for speed and VRAM efficiency (8GB limit)
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    
    # Setup ROS2 Reader to get synced frames
    reader = ROS2Reader(
        str(bag_path),
        num_frames=args.max_frames if args.max_frames > 0 else -1,
        nth_frames=args.nth_frames
    )
    
    from rosbags.highlevel import AnyReader
    
    print(f"[DAV2] Processing {len(reader.frames)} frames...")
    
    with AnyReader([bag_path], default_typestore=reader.typestore) as r:
        for i, frame in enumerate(tqdm(reader.frames)):
            # 1. Get Color and Depth
            msg_color = r.deserialize(frame["color_raw"], frame["color_conn"].msgtype)
            # ROS2 image to RGB
            img_raw = np.frombuffer(msg_color.data, dtype=np.uint8).reshape(msg_color.height, msg_color.width, 3)
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            
            msg_depth = r.deserialize(frame["depth_raw"], frame["depth_conn"].msgtype)
            depth_astra = np.frombuffer(msg_depth.data, dtype=np.uint16).reshape(msg_depth.height, msg_depth.width).astype(np.float32) / 1000.0
            
            # 2. Run Inference
            inputs = processor(images=img_rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            
            # 3. Align with Astra (Metric Scale)
            # Depth Anything is relative, we make it metric using Astra
            metric_depth = align_depth(prediction, depth_astra)
            
            # 4. Save (as uint16 PNG in millimeters for compatibility)
            depth_mm = (metric_depth * 1000).astype(np.uint16)
            save_path = output_dir / f"frame_{frame['timestamp']}.png"
            cv2.imwrite(str(save_path), depth_mm)

    print(f"\n[DAV2] Done! High-quality depths saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", type=str)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--nth_frames", type=int, default=1)
    args = parser.parse_args()
    main(args)
