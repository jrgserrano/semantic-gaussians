import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import tabulate

from opensplat3d.utils.setup_utils import setup_from_config, get_latest_model
from opensplat3d.eval.eval_lerf_ovs import get_annotations
from opensplat3d.language import LanguageModel
from opensplat3d.gaussian_renderer import render
from opensplat3d.params import PipeParams

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Labeling Accuracy (VLM vs GT)")
    parser.add_argument("model_dir", type=Path, help="Path to the trained model directory")
    parser.add_argument("--label_path", type=str, default=None, help="Path to LERF-OVS labels")
    parser.add_argument("--lang_model", type=str, default="masqclip", help="Language model for embeddings")
    args = parser.parse_args()

    # 1. Setup paths and env
    model_path = args.model_dir
    if not (model_path / "config.yaml").exists():
        model_path = get_latest_model(model_path)
    
    label_path = args.label_path
    label_path = Path(label_path)

    # 2. Load model and data
    from opensplat3d.config import Config
    config = Config.load(model_path / "config.yaml")
    setup_params = setup_from_config(config, model_path)
    
    scene_name = model_path.parent.parent.name
    scene_label_path = label_path / scene_name
    if not scene_label_path.exists():
        print(f"Error: Label path {scene_label_path} does not exist.")
        return

    gt_annotations, query_prompts = get_annotations(scene_label_path)
    
    # Load VLM descriptions from instances.json
    desc_path = model_path / "instances.json"
    if not desc_path.exists():
        print(f"Error: Descriptions not found at {desc_path}. Ensure the LVLM has generated instances.json.")
        return
    with open(desc_path, "r") as f:
        instances_data = json.load(f)
    
    # Map cluster ID to description
    vlm_descriptions = {}
    
    # Target the 'instances' key in your JSON
    instances_root = instances_data.get("instances", {})
    
    for key, info in instances_root.items():
        # 1. Extract numeric ID from keys like "obj0", "panel_12", "slab_13"
        import re
        match = re.search(r'(\d+)', key)
        if match:
            lid = match.group(1)
        else:
            lid = key # fallback to full key
            
        # 2. Get the description from the 'results' dict
        results = info.get("results", {})
        if results:
            # Take the first key that isn't 'unknown' if possible, or just the first one
            descs = [d for d in results.keys() if d != "unknown"]
            if descs:
                vlm_descriptions[str(lid)] = descs[0]
            else:
                vlm_descriptions[str(lid)] = list(results.keys())[0]

    # Load cluster labels for points
    cluster_labels = torch.from_numpy(np.load(model_path / "clustering" / "labels.npy")).to(setup_params.device)
    
    # 3. Load Language Model for Semantic Similarity
    lang_model = LanguageModel(args.lang_model)
    
    # Pre-compute embeddings for GT prompts and VLM descriptions
    print(f"[Eval] Computing embeddings for similarity comparison...")
    gt_embeddings = lang_model.embed_text(query_prompts, normalize=True) # [N_gt, D]
    
    vlm_labels_ids = sorted([int(k) for k in vlm_descriptions.keys()])
    vlm_texts = [vlm_descriptions[str(k)] for k in vlm_labels_ids]
    vlm_embeddings = lang_model.embed_text(vlm_texts, normalize=True) # [N_vlm, D]
    
    # Map back for easy access
    vlm_id_to_embed = {str(lid): vlm_embeddings[i] for i, lid in enumerate(vlm_labels_ids)}

    # 4. Matching and Scoring
    results = [] # List of (GT_label, VLM_label, Similarity)
    
    test_cameras = setup_params.scene.get_test_cameras()
    test_frame_names = [x.frame.stem for x in gt_annotations]
    
    # Filter cameras to match test frames
    test_cameras = [c for c in test_cameras if c.name in test_frame_names]

    print(f"[Eval] Matching clusters to GT objects across {len(test_cameras)} frames...")
    
    for cam in tqdm(test_cameras):
        # Find the annotation for this frame
        anno = next(a for a in gt_annotations if a.frame.stem == cam.name)
        
        # Render cluster IDs to this camera
        # We need to know which cluster dominates each pixel
        # Simplified: Render the 1D cluster labels
        render_pkg = render(
            cam, setup_params.gaussians, PipeParams(), 
            torch.zeros(3, device=setup_params.device), 
            setup_params.gaussians.active_sh_degree, 
            render_features=True
        )
        rendered_features = render_pkg.features.permute(1, 2, 0) # [H, W, C]
        H, W, C = rendered_features.shape
        
        # Get cluster prototypes (mean features)
        unique_labels = torch.unique(cluster_labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        protos = []
        valid_lids = []
        for lid in unique_labels:
            if str(lid.item()) in vlm_descriptions:
                mask = cluster_labels == lid
                protos.append(setup_params.gaussians.get_features[mask, :C].mean(dim=0))
                valid_lids.append(lid.item())
        
        if not protos: continue
        protos = torch.stack(protos)
        
        # Classify pixels to find clusters
        flat_features = rendered_features.reshape(-1, C)
        dists = torch.cdist(flat_features.unsqueeze(0), protos.unsqueeze(0)).squeeze(0)
        pred_labels_idx = torch.argmin(dists, dim=1)
        pred_labels = torch.tensor(valid_lids, device=setup_params.device)[pred_labels_idx].reshape(H, W)

        # For each GT object in this frame
        for gt_name, obj_anno in anno.annotations.items():
            gt_mask = torch.from_numpy(obj_anno.masks.sum(axis=0)).to(setup_params.device)
            
            # Find which cluster ID is most frequent in the GT mask area
            masked_preds = pred_labels[gt_mask]
            if masked_preds.numel() == 0: continue
            
            # Get the winning cluster ID
            vals, counts = torch.unique(masked_preds, return_counts=True)
            best_cluster_id = str(vals[torch.argmax(counts)].item())
            
            if best_cluster_id in vlm_descriptions:
                vlm_text = vlm_descriptions[best_cluster_id]
                
                # Compute Similarity
                gt_idx = query_prompts.index(gt_name)
                gt_emb = gt_embeddings[gt_idx]
                vlm_emb = vlm_id_to_embed[best_cluster_id]
                
                sim = torch.dot(gt_emb, vlm_emb).item()
                results.append({
                    "gt": gt_name,
                    "vlm": vlm_text,
                    "sim": sim,
                    "frame": cam.name
                })

    # 5. Report
    print("\n" + "="*50)
    print(f"SEMANTIC LABELING EVALUATION: {scene_name}")
    print("="*50)
    
    # Average by category
    cat_results = {}
    for r in results:
        if r["gt"] not in cat_results:
            cat_results[r["gt"]] = []
        cat_results[r["gt"]].append(r["sim"])
    
    table = [["Category", "Avg Similarity", "Top Example"]]
    all_sims = []
    for cat, sims in cat_results.items():
        avg_sim = np.mean(sims)
        all_sims.append(avg_sim)
        # Find an example with high similarity
        examples = [r for r in results if r["gt"] == cat]
        best_example = max(examples, key=lambda x: x["sim"])
        table.append([cat, f"{avg_sim:.4f}", f"'{best_example['vlm']}'"])
    
    table.append([tabulate.SEPARATING_LINE, "", ""])
    table.append(["MEAN SCORE", f"{np.mean(all_sims):.4f}", ""])
    
    print(tabulate.tabulate(table, headers="firstrow"))

if __name__ == "__main__":
    main()
