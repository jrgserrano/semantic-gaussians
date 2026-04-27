
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
import numpy as np

from pathlib import Path
bag_path = Path("/home/jsm15/datasets/astra_lab/astra_lab_0.db3")
typestore = get_typestore(Stores.ROS2_HUMBLE)

with AnyReader([bag_path], default_typestore=typestore) as reader:
    print(f"Topics in bag:")
    for connection in reader.connections:
        print(f"  {connection.topic} [{connection.msgtype}]")
    
    tf_msgs = 0
    tf_static_msgs = 0
    tf_topics = ["/tf", "/tf_static"]
    
    connections = [c for c in reader.connections if c.topic in tf_topics]
    frame_counts = {}
    
    for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc="Scanning TF"):
        msg = reader.deserialize(rawdata, connection.msgtype)
        if connection.topic == "/tf":
            tf_msgs += 1
        else:
            tf_static_msgs += 1
            
        for transform in msg.transforms:
            frame_counts[transform.child_frame_id] = frame_counts.get(transform.child_frame_id, 0) + 1
            
    print(f"\nTF Summary:")
    print(f"  /tf messages: {tf_msgs}")
    print(f"  /tf_static messages: {tf_static_msgs}")
    print(f"  Frames found and message counts:")
    for frame, count in frame_counts.items():
        print(f"    {frame}: {count}")
