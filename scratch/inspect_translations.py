
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
from pathlib import Path
import numpy as np

bag_path = Path("/home/jsm15/datasets/astra_lab/astra_lab_0.db3")
typestore = get_typestore(Stores.ROS2_HUMBLE)

with AnyReader([bag_path], default_typestore=typestore) as reader:
    tf_topics = ["/tf"]
    connections = [c for c in reader.connections if c.topic in tf_topics]
    
    first_t = None
    last_t = None
    
    for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc="Scanning TF"):
        msg = reader.deserialize(rawdata, connection.msgtype)
        for transform in msg.transforms:
            if transform.child_frame_id == "base_footprint":
                t = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
                if first_t is None:
                    first_t = t
                last_t = t
                
    print(f"\nBase Footprint Translation (relative to {msg.transforms[0].header.frame_id if msg.transforms else 'unknown'}):")
    print(f"  First: {first_t}")
    print(f"  Last: {last_t}")
    print(f"  Delta: {np.linalg.norm(last_t - first_t) if first_t is not None else 0:.4f}m")
