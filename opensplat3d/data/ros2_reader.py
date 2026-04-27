
import torch
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import cv2
from typing import List, Dict
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp
from opensplat3d.utils.scene_utils import CameraInfo

class ROS2Reader:
    def __init__(
        self,
        bag_path: str,
        color_topic: str = "/astra_camera/color/image_raw",
        depth_topic: str = "/astra_camera/depth/image_raw",
        tf_topic: str = "/tf",
        world_frame: str = "map",
        camera_frame: str = "astra_camera_color_optical_frame",
        intrinsics: Dict = None,
        num_frames: int = -1,
        nth_frames: int = -1,
        mask_subdir: str = None,
    ):
        self.bag_path = Path(bag_path)
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.tf_topic = tf_topic
        self.world_frame = world_frame
        self.camera_frame = camera_frame
        self.intrinsics = intrinsics
        self.mask_subdir = mask_subdir
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        
        self.frames = []
        self._index_bag(num_frames, nth_frames)

    def _index_bag(self, num_frames, nth_frames):
        print(f"[ROS2Reader] Indexing bag: {self.bag_path}")
        
        # Pass 1: Collect metadata and TF
        color_all_ts = []
        depth_all_ts = []
        tf_tree = {} # frame_id -> list of (timestamp, parent, pos, quat)
        
        with AnyReader([self.bag_path], default_typestore=self.typestore) as reader:
            connections = [c for c in reader.connections if c.topic in [self.color_topic, self.depth_topic, self.tf_topic, "/tf_static"]]
            for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc="Scanning Bag (Metadata)"):
                if connection.topic == self.tf_topic or connection.topic == "/tf_static":
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    for transform in msg.transforms:
                        tf_tree.setdefault(transform.child_frame_id, []).append((
                            timestamp, 
                            transform.header.frame_id, 
                            np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]),
                            np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
                        ))
                elif connection.topic == self.color_topic:
                    color_all_ts.append(timestamp)
                elif connection.topic == self.depth_topic:
                    depth_all_ts.append(timestamp)

        if not color_all_ts:
            print("[ROS2Reader] ERROR: No color images found!")
            return
            
        print(f"[ROS2Reader] Timestamp ranges:")
        print(f"  Color: {min(color_all_ts)} to {max(color_all_ts)} (Duration: {(max(color_all_ts)-min(color_all_ts))/1e9:.2f}s)")
        if depth_all_ts:
            print(f"  Depth: {min(depth_all_ts)} to {max(depth_all_ts)} (Duration: {(max(depth_all_ts)-min(depth_all_ts))/1e9:.2f}s)")
        
        for frame_id, msgs in tf_tree.items():
            t_min = min(m[0] for m in msgs)
            t_max = max(m[0] for m in msgs)
            if t_max < min(color_all_ts) or t_min > max(color_all_ts):
                print(f"  [!] TF Frame '{frame_id}' DOES NOT OVERLAP with color images!")
                print(f"      TF: {t_min} to {t_max}")

        # Patch broken TF chain: Connect camera_link to astra_camera_link
        if "camera_link" not in tf_tree:
            print("[ROS2Reader] Patching TF: Connecting camera_link -> astra_camera_link (Identity)")
            tf_tree["camera_link"] = [(0, "astra_camera_link", np.zeros(3), np.array([0, 0, 0, 1]))]

        # Select frames and find synced depth
        selected_color_ts = []
        skipped_no_tf = 0
        for i, ts in enumerate(color_all_ts):
            if nth_frames > 0 and i % nth_frames != 0: continue
            
            # Find closest depth
            if not depth_all_ts: continue
            d_ts = min(depth_all_ts, key=lambda x: abs(x - ts))
            if abs(d_ts - ts) > 2e8: continue # 200ms threshold for depth
            
            # Strict Pose check (no extrapolation beyond 100ms)
            pose = self._get_pose(ts, tf_tree, strict=True)
            if pose is None:
                skipped_no_tf += 1
                continue
            
            selected_color_ts.append((ts, d_ts, pose))
            if num_frames > 0 and len(selected_color_ts) >= num_frames: break
            
        if skipped_no_tf > 0:
            print(f"[ROS2Reader] Warning: Skipped {skipped_no_tf} frames due to missing/distant TF data.")

        # Pass 2: Extract data for selected frames
        color_ts_to_find = {x[0] for x in selected_color_ts}
        depth_ts_to_find = {x[1] for x in selected_color_ts}
        extracted_color = {}
        extracted_depth = {}
        
        with AnyReader([self.bag_path], default_typestore=self.typestore) as reader:
            connections = [c for c in reader.connections if c.topic in [self.color_topic, self.depth_topic]]
            for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc="Scanning Bag (Data)"):
                if connection.topic == self.color_topic and timestamp in color_ts_to_find:
                    extracted_color[timestamp] = (rawdata, connection)
                elif connection.topic == self.depth_topic and timestamp in depth_ts_to_find:
                    extracted_depth[timestamp] = (rawdata, connection)

        # Build final frames
        self.frames = []
        for ts, d_ts, pose in selected_color_ts:
            if ts in extracted_color and d_ts in extracted_depth:
                self.frames.append({
                    "timestamp": ts,
                    "color_raw": extracted_color[ts][0],
                    "color_conn": extracted_color[ts][1],
                    "depth_raw": extracted_depth[d_ts][0],
                    "depth_conn": extracted_depth[d_ts][1],
                    "pose": pose
                })
        
        print(f"[ROS2Reader] Successfully indexed {len(self.frames)} synced frames.")

    def _get_pose(self, timestamp, tf_tree, strict=False):
        # Very basic TF interpolation for map -> camera
        # In a real robot, we'd traverse the tree. Here we'll try to find direct path or base_link path
        
        # Target: camera_frame w.r.t world_frame
        # Let's look for world_frame -> base_link and base_link -> camera_frame
        def interpolate(frames, ts, strict=False):
            if not frames: return None
            # Find closest frame
            frames.sort(key=lambda x: x[0])
            times = [f[0] for f in frames]
            
            # Static TF check: If only one frame, it's valid for all time
            if len(frames) == 1:
                return frames[0][2], frames[0][3]
                
            idx = np.searchsorted(times, ts)
            
            # Check bounds for strict mode (only for dynamic TFs)
            if strict:
                closest_t = times[0] if idx == 0 else (times[-1] if idx == len(times) else times[idx])
                if abs(ts - closest_t) > 2e8: # 200ms for dynamic
                    return None

            if idx == 0: return frames[0][2], frames[0][3]
            if idx == len(frames): return frames[-1][2], frames[-1][3]
            
            # Interpolate between idx-1 and idx
            f0 = frames[idx-1]
            f1 = frames[idx]
            t0, t1 = f0[0], f1[0]
            alpha = (ts - t0) / (t1 - t0)
            
            pos = (1-alpha) * f0[2] + alpha * f1[2]
            # Slerp for rotation
            rotations = R.from_quat([f0[3], f1[3]])
            slerp = Slerp([t0, t1], rotations)
            res_r = slerp([ts])[0]
            
            return pos, res_r.as_quat()

        # Try to find a path to world_frame
        curr = self.camera_frame
        full_transform = np.eye(4)
        
        while curr != self.world_frame:
            if curr not in tf_tree:
                if len(self.frames) == 0: # Only print for the first frame to avoid spam
                    print(f"[ROS2Reader] TF Error: Frame '{curr}' not found in tree. Available: {list(tf_tree.keys())[:5]}...")
                return None
            
            # Interpolate this link
            res = interpolate(tf_tree[curr], timestamp, strict=strict)
            if res is None: return None
            pos, quat = res
            
            # Local matrix
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat(quat).as_matrix()
            mat[:3, 3] = pos
            
            full_transform = mat @ full_transform
            curr = tf_tree[curr][0][1] # Get parent from first msg
            
            if curr == "odom" and self.world_frame == "map":
                # Handle map->odom separately if needed
                pass

        # Rotate to 3DGS/OpenCV coordinate system if not already an optical frame
        # ROS Standard: Forward=X, Left=Y, Up=Z (FLU)
        # 3DGS/OpenCV: Right=X, Down=Y, Forward=Z (RDF)
        
        is_optical = "optical" in self.camera_frame.lower()
        
        R_world_camera = full_transform[:3, :3]
        if not is_optical:
            R_flu_to_rdf = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ])
            full_transform[:3, :3] = R_world_camera @ R_flu_to_rdf
        else:
            # Already RDF
            full_transform[:3, :3] = R_world_camera

        return full_transform

    def __len__(self):
        return len(self.frames)

    def load_train(self, progbar=False):
        print(f"[ROS2Reader] Loading {len(self.frames)} cameras from bag...")
        cameras = []
        with AnyReader([self.bag_path], default_typestore=self.typestore) as reader:
            for i in tqdm(range(len(self.frames)), desc="Loading Cameras", disable=not progbar):
                cameras.append(self.get_camera(i, reader=reader))
        return cameras

    def load_test(self, progbar=False):
        return []

    def get_camera(self, idx, device="cuda", reader=None):
        frame = self.frames[idx]
        
        # Internal helper to handle optional reader
        def _process(r):
            import cv2
            # Decode Color
            msg_color = r.deserialize(frame["color_raw"], frame["color_conn"].msgtype)
            msg_type = frame["color_conn"].msgtype
            
            if "CompressedImage" in msg_type:
                img = cv2.imdecode(np.frombuffer(msg_color.data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    print(f"[ROS2Reader] Error decoding compressed color image!")
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Raw Image
                h, w = msg_color.height, msg_color.width
                img = np.frombuffer(msg_color.data, dtype=np.uint8)
                # Handle potential flattening or weird steps
                channels = len(img) // (h * w) if (h * w) > 0 else 3
                try:
                    img = img.reshape(h, w, channels)
                    encoding = msg_color.encoding.lower() if hasattr(msg_color, 'encoding') else ''
                    if 'bgr' in encoding:
                        if channels == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        elif channels == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    elif 'rgb' in encoding:
                        if channels == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        # if channels == 3 and it's rgb, do nothing
                    else:
                        # Fallback
                        if channels == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        elif channels == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    print(f"[ROS2Reader] Failed reshape for raw image {h}x{w}x{channels}. Size: {len(img)}")
                    img = np.zeros((480, 640, 3), dtype=np.uint8)

            # Apply Brightness/Gamma Correction if images are "apagadas"
            # Gamma > 1.0 brightens the image. 1.2-1.5 is usually a good range.
            gamma = 1.75
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

            # Decode Depth
            dav2_path = self.bag_path.parent / "depth_dav2" / f"frame_{frame['timestamp']}.png"
            if dav2_path.exists():
                depth = cv2.imread(str(dav2_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            else:
                msg_depth = r.deserialize(frame["depth_raw"], frame["depth_conn"].msgtype)
                depth_type = frame["depth_conn"].msgtype
                
                if "CompressedImage" in depth_type:
                    depth_img = cv2.imdecode(np.frombuffer(msg_depth.data, np.uint8), cv2.IMREAD_UNCHANGED)
                    if depth_img is not None:
                        depth = depth_img.astype(np.float32) / 1000.0
                    else:
                        depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                else:
                    h_d, w_d = msg_depth.height, msg_depth.width
                    depth = np.frombuffer(msg_depth.data, dtype=np.uint16)
                    try:
                        depth = depth.reshape(h_d, w_d).astype(np.float32) / 1000.0
                    except:
                        print(f"[ROS2Reader] Failed reshape for depth {h_d}x{w_d}. Size: {len(depth)}")
                        depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

            if not hasattr(self, "_decoded_once"):
                print(f"[ROS2Reader] First Decoded Frame: Color {img.shape} | Depth {depth.shape}")
                self._decoded_once = True
            
            return img, depth, img.shape[1], img.shape[0]

        if reader is None:
            with AnyReader([self.bag_path], default_typestore=self.typestore) as r:
                img, depth, width, height = _process(r)
        else:
            img, depth, width, height = _process(reader)
            
        # Convert Pose to Gaussian Splatting format
        # ROS: Forward=X, Left=Y, Up=Z
        # GS: Right=X, Down=Y, Forward=Z
        
        # c2w is in World coordinate system (Camera to World)
        c2w = frame["pose"]
        
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]
        
        # The renderer expects R to be C2W rotation (it will be transposed to W2C internally)
        # But it expects T to be the W2C translation (X_cam = R_w2c * X_world + T_w2c)
        R_w2c = R_c2w.T
        T_w2c = -R_w2c @ T_c2w
        
        # Intrinsics
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]
        
        # Load Masks if available
        masks = None
        if self.mask_subdir:
            mask_dir = self.bag_path.parent / self.bag_path.stem / self.mask_subdir
            mask_path_npz = mask_dir / f"frame_{idx}.npz"
            mask_path_png = mask_dir / f"frame_{idx}.png"
            
            if mask_path_npz.exists():
                with np.load(mask_path_npz) as level_masks:
                    masks = torch.from_numpy(level_masks["default"]).long()
            elif mask_path_png.exists():
                mask_img = cv2.imread(str(mask_path_png), cv2.IMREAD_UNCHANGED)
                masks = torch.from_numpy(mask_img.copy()).long()
        
        return CameraInfo(
            uid=idx,
            R=torch.from_numpy(R_c2w.copy()).float(),
            T=torch.from_numpy(T_w2c.copy()).float(),
            fovY=2 * np.arctan(height / (2 * fy)),
            fovX=2 * np.arctan(width / (2 * fx)),
            image=torch.from_numpy(img.copy()), # HWC uint8
            image_path=Path(f"bag_{idx}"),
            image_name=f"frame_{idx}",
            width=width,
            height=height,
            masks=masks,
            depth=torch.from_numpy(depth.copy()).to(device) if depth is not None else None
        )
