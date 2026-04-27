
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
from pathlib import Path

bag_path = Path("/home/jsm15/datasets/astra_lab/astra_lab_0.db3")
typestore = get_typestore(Stores.ROS2_HUMBLE)

with AnyReader([bag_path], default_typestore=typestore) as reader:
    tf_topics = ["/tf", "/tf_static"]
    connections = [c for c in reader.connections if c.topic in tf_topics]
    
    ranges = {} # frame -> (min, max)
    
    for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc="Scanning TF"):
        msg = reader.deserialize(rawdata, connection.msgtype)
        for transform in msg.transforms:
            f = transform.child_frame_id
            t = timestamp
            if f not in ranges:
                ranges[f] = [t, t]
            else:
                ranges[f][0] = min(ranges[f][0], t)
                ranges[f][1] = max(ranges[f][1], t)
            
    print(f"\nTF Ranges:")
    for f, (t_min, t_max) in ranges.items():
        print(f"  {f}: {t_min} to {t_max} (Duration: {(t_max-t_min)/1e9:.2f}s)")
