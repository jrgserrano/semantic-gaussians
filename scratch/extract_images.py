
import os
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np
from tqdm import tqdm

import argparse

def extract_images(bag_path, out_dir, nth=1, max_frames=0):
    bag_path = Path(bag_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    print(f"Extracting images from {bag_path} to {out_dir}...")
    
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        # Find color topic
        color_conn = [c for c in reader.connections if 'color' in c.topic][0]
        
        count = 0
        saved = 0
        for connection, timestamp, rawdata in tqdm(reader.messages(connections=[color_conn])):
            if max_frames > 0 and saved >= max_frames:
                break
            if count % nth == 0:
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                # Save as BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / f"frame_{count}.png"), img_bgr)
                saved += 1
            count += 1
    
    print(f"Done! Extracted {saved} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", type=str, default="/home/jsm15/datasets/astra_lab/astra_lab_0.db3", nargs='?')
    parser.add_argument("--out_dir", type=str, default="/home/jsm15/datasets/astra_lab/astra_lab_0/images")
    parser.add_argument("--nth_frames", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()
    
    extract_images(args.bag_path, args.out_dir, nth=args.nth_frames, max_frames=args.max_frames)
