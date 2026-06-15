import os
import sys
import json
import argparse
import random
import urllib.parse
from pathlib import Path
import numpy as np
import torch

from core.utils.setup_utils import setup
from core.gaussian_model import create_from_ply, create_from_pcd
from core.utils.scene_utils import fetch_ply
from plyfile import PlyData
from core.viewer.server import ViewerServer

def run_viewer(model_dir=None, ply=None, port=8080, sh_degree=3, agent_mode=False):
    print("[Viewer] Starting run_viewer...")
    if model_dir is None and ply is None:
        print("[Error] You must provide either model_dir or ply path.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Viewer] Running on device: {device}")
    
    # 1. Resolve paths
    ply_path = None
    label_path = None
    instances_path = None
    descriptions_path = None
    cameras = []
    
    if model_dir:
        model_path = Path(model_dir)
        if not model_path.exists():
            print(f"[Error] Model directory not found: {model_path}")
            return
            
        print(f"[Viewer] Setting up model from directory: {model_path}")
        try:
            print("[Viewer] Calling setup_utils.setup()...")
            setup_params = setup(model_path, load_masks=False)
            print("[Viewer] setup() finished successfully.")
            gaussians = setup_params.gaussians
            cameras = setup_params.scene.get_train_cameras()
            
            # Find associated semantic/instance files
            label_path = setup_params.model_path / "clustering" / "labels.npy"
            instances_path = setup_params.model_path / "instances.json"
            descriptions_path = setup_params.model_path / "descriptions.pth"
        except Exception as e:
            print(f"[Warning] Setup helper failed ({e}). Attempting manual resolution...")
            # Fallback to manual folder search
            point_cloud_dir = model_path / "point_cloud"
            if point_cloud_dir.exists():
                iterations = sorted([int(p.name.split("_")[-1]) for p in point_cloud_dir.iterdir() if p.is_dir()])
                if iterations:
                    latest = iterations[-1]
                    ply_path = point_cloud_dir / f"iteration_{latest}" / "point_cloud.ply"
            
            label_path = model_path / "clustering" / "labels.npy"
            instances_path = model_path / "instances.json"
            descriptions_path = model_path / "descriptions.pth"
            
            if ply_path and ply_path.exists():
                print(f"[Viewer] Found PLY file: {ply_path}")
                gaussians = create_from_ply(str(ply_path), sh_degree, device)
            else:
                print(f"[Error] Could not locate point_cloud.ply in {model_path}")
                return
    else:
        # Load from direct PLY path
        ply_path = Path(ply)
        if not ply_path.exists():
            print(f"[Error] PLY file not found: {ply_path}")
            return
            
        print(f"[Viewer] Loading PLY file directly: {ply_path}")
        
        # Check if it's a Gaussian splat PLY or a raw point cloud
        try:
            plydata = PlyData.read(str(ply_path))
            prop_names = [p.name for p in plydata.elements[0].properties]
            if "opacity" not in prop_names:
                print(f"[Viewer] PLY lacks 'opacity'. Loading as raw point cloud...")
                pcd, _ = fetch_ply(ply_path)
                gaussians = create_from_pcd(pcd, sh_degree, device)
                # Force very small scales so they look like normal points instead of blobs
                gaussians._scaling.data.fill_(-7.0)
            else:
                gaussians = create_from_ply(str(ply_path), sh_degree, device)
        except Exception as e:
            print(f"[Error] Failed to load PLY: {e}")
            return
        
        # Look for labels next to the ply file or in its parent folders
        parent = ply_path.parent
        label_path = parent / "labels.npy"
        if not label_path.exists():
            label_path = parent.parent / "clustering" / "labels.npy"
            instances_path = parent.parent / "instances.json"
            descriptions_path = parent.parent / "descriptions.pth"
            
    # 2. Load semantic labels
    labels = None
    if label_path and label_path.exists():
        print(f"[Viewer] Loading semantic labels from: {label_path}")
        labels = np.load(label_path)
        assert len(labels) == gaussians.num_points, "[Error] PLY point count and labels count mismatch."
    else:
        print("[Viewer] No labels.npy found. Semantic features will not be active (RGB mode only).")

    # 3. Load descriptions / labels text (etiquetas)
    vlm_descriptions = {}
    vlm_full_descriptions = {}
    if labels is not None:
        # Try instances.json
        if instances_path and instances_path.exists():
            try:
                print(f"[Viewer] Loading instances from: {instances_path}")
                with open(instances_path, "r") as f:
                    inst_data = json.load(f)
                import re
                for obj_key, info in inst_data.get("instances", {}).items():
                    id_match = re.search(r'(\d+)', obj_key)
                    if id_match:
                        lbl_id = int(id_match.group(1))
                        label_match = re.match(r'([a-zA-Z]+)', obj_key)
                        clean_label = label_match.group(1) if label_match else obj_key
                        vlm_descriptions[lbl_id] = clean_label
                        vlm_full_descriptions[lbl_id] = clean_label
            except Exception as e:
                print(f"[Warning] Error parsing instances.json: {e}")

        # Always try to load full descriptions from descriptions.pth if available
        if descriptions_path and descriptions_path.exists():
            try:
                print(f"[Viewer] Loading descriptions from: {descriptions_path}")
                desc_data = torch.load(descriptions_path, map_location="cpu", weights_only=False)
                for k, v in desc_data.items():
                    if k not in vlm_descriptions:
                        vlm_descriptions[k] = v.get("identifier", v["description"])
                    # Always set the full description if available
                    vlm_full_descriptions[k] = v.get("description", vlm_full_descriptions.get(k, ""))
            except Exception as e:
                print(f"[Warning] Error parsing descriptions.pth: {e}")

    # 3.5. Load or dynamically generate bounding boxes
    bboxes = {}
    if labels is not None:
        if model_dir:
            bbox_path = Path(model_dir) / "clustering" / "bboxes.pth"
        else:
            bbox_path = ply_path.parent / "bboxes.pth"
            if not bbox_path.exists():
                bbox_path = ply_path.parent.parent / "clustering" / "bboxes.pth"
        
        if not bbox_path.exists():
            print(f"[Viewer] Bounding boxes not found at {bbox_path}. Generating them automatically (fast mode)...")
            try:
                from core.semantic.generate_bboxes import generate_bboxes_from_data
                xyz = gaussians.get_xyz.detach().cpu().numpy()
                bboxes = generate_bboxes_from_data(xyz, labels, bbox_path)
            except Exception as e:
                print(f"[Viewer] Failed to generate bounding boxes automatically: {e}")

        if bbox_path.exists():
            print(f"[Viewer] Loading bounding boxes from: {bbox_path}")
            bboxes = torch.load(bbox_path, weights_only=False)

    # 4. Generate random distinct colors for each unique label ID
    label_colors = {}
    if labels is not None:
        unique_labels = np.unique(labels)
        random.seed(42)
        for lbl in unique_labels:
            lbl = int(lbl)
            if lbl == -1:
                label_colors[lbl] = np.array([15, 15, 15], dtype=np.uint8) # Unlabeled is dark gray
            else:
                # Generate vivid harmonious colors
                label_colors[lbl] = np.array([
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                ], dtype=np.uint8)

    # 4.5. Compute 3D centroids for each unique label ID (for floating HTML labels)
    centroids = {}
    if labels is not None:
        print("[Viewer] Computing 3D centroids for all labels...")
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        for lbl in unique_labels:
            lbl = int(lbl)
            mask = (labels == lbl)
            if mask.sum() > 0:
                centroids[lbl] = xyz[mask].mean(axis=0).tolist()

    # 5. Pack cameras/poses JSON to follow
    poses_data = {
        "current_pose": None,
        "keyframes": []
    }
    if cameras:
        keyframes_list = []
        for cam in cameras:
            # cam.R is the transposed w2c rotation matrix, and cam.T is the w2c translation
            # We reconstruct the world-to-camera matrix (w2c)
            w2c = np.eye(4)
            w2c[:3, :3] = cam.R.detach().cpu().numpy().T
            w2c[:3, 3] = cam.T.detach().cpu().numpy()
            
            # Invert w2c to get the correct camera-to-world (c2w) matrix in physical world coordinates
            c2w = np.linalg.inv(w2c)
            keyframes_list.append(c2w.T.flatten().tolist())
            
        poses_data = {
            "current_pose": keyframes_list[0] if keyframes_list else None,
            "keyframes": keyframes_list
        }

    # 6. Define dynamic WebGL splat packing function
    def pack_gaussians(mode="rgb", select_id="all"):
        xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
        scaling = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
        rotation = gaussians.get_rotation.detach().cpu().numpy().astype(np.float32)
        opacity = gaussians.get_opacity.detach().cpu().numpy().astype(np.float32)
        
        # Determine colors
        if mode == "labels" and labels is not None:
            colors = np.zeros((len(xyz), 3), dtype=np.float32)
            for lbl, color in label_colors.items():
                mask = (labels == lbl)
                colors[mask] = color / 255.0
        else:
            # Default RGB mode
            shs = gaussians._features_dc.detach().cpu()
            if shs.ndim == 3:
                shs = shs.squeeze(1)
            colors = (shs * 0.28209479177387814 + 0.5).clamp(0.0, 1.0).numpy().astype(np.float32)
        
        # Apply selection isolation
        if select_id != "all" and labels is not None:
            try:
                target_id = int(select_id)
                mask = (labels != target_id)
                if agent_mode:
                    colors = colors.copy()
                    colors[mask] = colors[mask] * 0.2
                else:
                    opacity = opacity.copy()
                    opacity[mask] = 0.0
            except ValueError:
                pass

        # Pack to standard 32-byte WebGL structure
        num_gaussians = xyz.shape[0]
        splat_data = np.zeros(num_gaussians, dtype=[
            ('position', 'f4', 3),
            ('scales', 'f4', 3),
            ('rgba', 'u1', 4),
            ('rot', 'u1', 4)
        ])
        
        splat_data['position'] = xyz
        splat_data['scales'] = scaling
        
        splat_data['rgba'][:, 0] = np.clip(colors[:, 0] * 255.0, 0, 255).astype(np.uint8)
        splat_data['rgba'][:, 1] = np.clip(colors[:, 1] * 255.0, 0, 255).astype(np.uint8)
        splat_data['rgba'][:, 2] = np.clip(colors[:, 2] * 255.0, 0, 255).astype(np.uint8)
        splat_data['rgba'][:, 3] = np.clip(opacity[:, 0] * 255.0, 0, 255).astype(np.uint8)
        
        # Normalize rotations
        rot_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
        rot_norm[rot_norm == 0] = 1.0
        norm_rotation = rotation / rot_norm
        splat_data['rot'] = np.clip(norm_rotation * 128.0 + 128.0, 0, 255).astype(np.uint8)
        
        return splat_data.tobytes()

    # 7. Start the WebGL Server and set callbacks
    server = ViewerServer(port=port)
    server.poses_data = poses_data
    server.config_data = {
        "is_training": False,
        "has_labels": labels is not None,
        "has_bboxes": len(bboxes) > 0,
        "agent_mode": agent_mode
    }
    
    last_requested = {"mode": None, "id": None}
    
    # Callback to handle dynamic /gaussians queries
    def gaussians_callback(path_with_query):
        parsed = urllib.parse.urlparse(path_with_query)
        params = urllib.parse.parse_qs(parsed.query)
        
        mode = params.get("mode", ["rgb"])[0]
        select_id = params.get("id", ["all"])[0]
        
        # Only print when the state actually changes to avoid terminal spam
        if last_requested["mode"] != mode or last_requested["id"] != select_id:
            last_requested["mode"] = mode
            last_requested["id"] = select_id
            print(f"[ViewerServer GET] /gaussians | Mode: {mode.upper()} | Selected ID: {select_id}")
            
        return pack_gaussians(mode, select_id)
        
    # Callback to handle /labels metadata
    def labels_callback():
        labels_list = []
        if labels is not None:
            # Sort label IDs numerically
            unique_ids = sorted([int(lbl) for lbl in label_colors.keys()])
            for lbl_id in unique_ids:
                desc = vlm_descriptions.get(lbl_id, f"Objeto #{lbl_id}")
                full_desc = vlm_full_descriptions.get(lbl_id, "")
                if lbl_id == -1:
                    desc = "Sin Etiqueta (Fondo)"
                    full_desc = "Fondo del escenario sin etiqueta semántica."
                color_list = label_colors[lbl_id].tolist()
                centroid = centroids.get(lbl_id, [0.0, 0.0, 0.0])
                bbox = bboxes.get(lbl_id, None)
                labels_list.append({
                    "id": lbl_id,
                    "name": f"#{lbl_id}: {desc}",
                    "description": full_desc,
                    "color": color_list,
                    "center": centroid,
                    "bbox": bbox
                })
        
        res = {"labels": labels_list}
        return json.dumps(res).encode("utf-8")

    server.gaussians_callback = gaussians_callback
    server.labels_callback = labels_callback
    server.start()

    # Beautiful interactive console UI
    print("=" * 70)
    print("       WEBGL SEMANTIC 3D GAUSSIAN SPLAT VIEWER DEPLOYED SUCCESSFULLY!")
    print("=" * 70)
    print(f" -> Local Interactive URL:  http://localhost:{port}")
    if labels is not None:
        print(f" -> Loaded Gaussian splats: {gaussians.num_points:,}")
        print(f" -> Segmented Instances:    {len(label_colors) - 1 if -1 in label_colors else len(label_colors)}")
    else:
        print(f" -> Loaded Gaussian splats: {gaussians.num_points:,} (RGB Mode Only)")
    print("=" * 70)
    print(" Press Ctrl+C in this terminal to terminate the server at any time.")
    print("=" * 70)

    # Keep main thread alive
    try:
        import time
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Viewer] Shutting down visualizer server...")
        server.stop()
        print("[Viewer] Visualizer stopped. Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebGL Semantic 3DGS offline visualizer.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default=None, 
        help="Path to the trained model output directory (e.g. outputs/Replica/room0/)"
    )
    parser.add_argument(
        "--ply", 
        type=str, 
        default=None, 
        help="Direct path to a point_cloud.ply file"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Port to run the WebGL HTTP server (default: 8080)"
    )
    parser.add_argument(
        "--sh_degree", 
        type=int, 
        default=3, 
        help="SH degree for loading PLY (default: 3)"
    )
    parser.add_argument(
        "--agent_mode",
        action="store_true",
        help="Launch the visualizer in exclusive Agent Mode (no manual navigation, simplified UI)"
    )
    args = parser.parse_args()
    run_viewer(model_dir=args.model_dir, ply=args.ply, port=args.port, sh_degree=args.sh_degree, agent_mode=args.agent_mode)
