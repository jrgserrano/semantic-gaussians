
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
from pathlib import Path

bag_path = Path("/home/jsm15/datasets/astra_lab/astra_lab_0.db3")
typestore = get_typestore(Stores.ROS2_HUMBLE)

with AnyReader([bag_path], default_typestore=typestore) as reader:
    tf_topics = ["/tf", "/tf_static"]
    connections = [c for c in reader.connections if c.topic in tf_topics]
    
    tree = {} # child -> parent
    
    for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc="Scanning TF"):
        msg = reader.deserialize(rawdata, connection.msgtype)
        for transform in msg.transforms:
            tree[transform.child_frame_id] = transform.header.frame_id
            
    print(f"\nTF Tree Structure (child -> parent):")
    for child, parent in tree.items():
        print(f"  {child} -> {parent}")
