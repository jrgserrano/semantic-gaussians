import argparse
import os
from pathlib import Path

import numpy as np
import tabulate
import torch
import imageio.v3 as iio
from tqdm import tqdm
import gc

# core imports
from core.gaussian_renderer import render
from core.params import PipeParams
from core.utils.setup_utils import setup
from core.eval.metrics import calculate_iou, calculate_biou

def calculate_ioa_vectorized(pred_mask: torch.Tensor, gt_masks_stack: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoA (Intersection over Area of pred_mask) against a stack of GT masks.
    pred_mask: (H, W) boolean
    gt_masks_stack: (N, H, W) boolean
    Returns: (N,) float tensor of IoAs
    """
    intersection = (gt_masks_stack & pred_mask).sum(dim=(1, 2)).float()
    area_pred = pred_mask.sum().float()
    if area_pred == 0:
        return torch.zeros_like(intersection)
    return intersection / area_pred

def run_evaluation_logic(setup_params, labels, threshold=0.5, save_masks=False, merge_threshold=0.5, compute_biou=True, eval_all=False):
    """Core evaluation loop separated for reusability. Optimized for speed."""
    gaussians = setup_params.gaussians
    scene = setup_params.scene
    model_path = setup_params.model_path 
    device = gaussians._xyz.device
    
    unique_labels = torch.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    if eval_all:
        print(f"📚 Evaluating on ALL cameras (Train + Test)")
        test_cameras = scene.get_train_cameras() + scene.get_test_cameras()
    else:
        test_cameras = scene.get_test_cameras()
        if not test_cameras:
            test_cameras = scene.get_train_cameras()
    
    if save_masks:
        debug_dir = model_path / f"debug_masks_merged_iter_{setup_params.iteration}"
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    per_instance_stats = {} 
    
    for cam_idx, cam in enumerate(tqdm(test_cameras, desc="Evaluating views", leave=False)):
        if cam.masks is None:
            continue
            
        gt_mask = cam.masks.to(device)
        gt_instances = torch.unique(gt_mask)
        gt_instances = gt_instances[gt_instances != -1]
        
        if len(gt_instances) == 0:
            continue
            
        # We will accumulate merged masks for each GT instance on-the-fly to save RAM
        merged_masks = {} # gt_id -> merged_boolean_mask
        fragments_count = {} # gt_id -> count
        
        # Calculate GT masks once per frame - STACKED on GPU for vectorized math
        gt_ids = sorted([int(idx.item()) for idx in gt_instances])
        # Shape: (num_gt, H, W)
        gt_masks_stack = torch.stack([(gt_mask == idx) for idx in gt_ids])
        
        # Pre-allocate override color tensor to reuse it across clusters
        override_color = torch.zeros((gaussians.num_points, 3), device=device)
        
        for cluster_id in unique_labels:
            mask_gauss = (labels == cluster_id)
            override_color.zero_() 
            override_color[mask_gauss] = 1.0
            
            render_pkg = render(cam, gaussians, PipeParams(), torch.zeros(3, device=device), 0, override_color=override_color)
            
            # Predict mask on GPU
            pred_mask = (render_pkg.render.mean(dim=0) > threshold)
            
            # --- Vectorized Streaming Comparison ---
            # ioas shape: (num_gt,)
            ioas = calculate_ioa_vectorized(pred_mask, gt_masks_stack)
            intersecting_gt_indices = (ioas > merge_threshold).nonzero(as_tuple=True)[0]
            
            for idx_in_stack in intersecting_gt_indices:
                gt_id = gt_ids[idx_in_stack.item()]
                if gt_id not in merged_masks:
                    merged_masks[gt_id] = pred_mask.detach().clone()
                    fragments_count[gt_id] = 1
                else:
                    merged_masks[gt_id] |= pred_mask
                    fragments_count[gt_id] += 1
            
            # Immediate cleanup
            del render_pkg
            del pred_mask
            del ioas
        # Clear override color after all clusters for this frame are processed
        del override_color
        
        # Clear cache after processing all clusters for this view
        torch.cuda.empty_cache()
        gc.collect()
        
        # Final IoU calculation for this frame using the merged results
        for idx_in_stack, gt_id in enumerate(gt_ids):
            mask_gt = gt_masks_stack[idx_in_stack]
            if gt_id in merged_masks:
                mask_pred = merged_masks[gt_id]
                
                # GPU Metrics calculation
                intersection = (mask_gt & mask_pred).sum().float()
                union = (mask_gt | mask_pred).sum().float()
                area_gt = mask_gt.sum().float()
                
                iou = (intersection / union).item() if union > 0 else 0.0
                acc = (intersection / area_gt).item() if area_gt > 0 else 0.0
                
                # BIoU is very slow (CPU-based), skip it during parameter sweeps if requested
                if compute_biou:
                    biou = calculate_biou(mask_gt.cpu().numpy(), mask_pred.cpu().numpy())
                else:
                    biou = 0.0
                    
                merged_count = fragments_count[gt_id]
                del mask_pred 
            else:
                iou = 0.0
                acc = 0.0
                biou = 0.0
                merged_count = 0
            
            if gt_id not in per_instance_stats:
                per_instance_stats[gt_id] = {"ious": [], "accs": [], "bious": [], "fragments": []}
            per_instance_stats[gt_id]["ious"].append(iou)
            per_instance_stats[gt_id]["accs"].append(acc)
            per_instance_stats[gt_id]["bious"].append(biou)
            per_instance_stats[gt_id]["fragments"].append(merged_count)
            
        # Final cleanup for this frame
        del gt_masks_stack
        del merged_masks
        del fragments_count
                
    all_mious = [np.mean(inst["ious"]) for inst in per_instance_stats.values()]
    all_maccs = [np.mean(inst["accs"]) for inst in per_instance_stats.values()]
    all_mbious = [np.mean(inst["bious"]) for inst in per_instance_stats.values()]
    final_miou = np.mean(all_mious) if all_mious else 0.0
    final_macc = np.mean(all_maccs) if all_maccs else 0.0
    final_mbiou = np.mean(all_mbious) if all_mbious else 0.0
    
    return final_miou, final_macc, final_mbiou, per_instance_stats

@torch.no_grad()
def evaluate_segmentation(model_path: Path, iteration: int | None = None, threshold: float = 0.5, device: str = "cuda:0", save_masks: bool = False, merge_threshold: float = 0.5, eval_all: bool = False):
    """
    Evaluates segmentation quality with Greedy Cluster Merging.
    """
    print(f"\nStarting evaluation for: {model_path}")
    
    setup_params = setup(model_path, iteration=iteration)
    
    labels_path = model_path / "clustering" / "labels.npy"
    if not labels_path.exists():
        print(f"Error: Labels not found at {labels_path}.")
        return
    
    labels = torch.from_numpy(np.load(labels_path)).to(setup_params.device)
    
    final_miou, final_macc, final_mbiou, per_instance_stats = run_evaluation_logic(
        setup_params, labels, threshold, save_masks, merge_threshold, compute_biou=True, eval_all=eval_all
    )
    
    table_data = []
    for gt_id, stats in sorted(per_instance_stats.items()):
        table_data.append([
            f"Instance {gt_id}", 
            f"{np.mean(stats['ious']):.4f}", 
            f"{np.mean(stats['accs']):.4f}",
            f"{np.mean(stats['bious']):.4f}",
            f"{np.mean(stats['fragments']):.1f}"
        ])
        
    report = "\n" + "="*80 + "\n"
    report += " SEGMENTATION EVALUATION REPORT\n"
    report += "="*80 + "\n"
    report += tabulate.tabulate(table_data, headers=["Instance ID", "IoU", "Accuracy", "BIoU", "Avg Frag"], tablefmt="grid")
    report += "\n" + "-"*80 + "\n"
    report += f"mIoU:  {final_miou:.4f}\n"
    report += f"mAcc:  {final_macc:.4f}\n"
    report += f"mBIoU: {final_mbiou:.4f}\n"
    report += "-"*70 + "\n"
    report += f"Note: Evaluation allows multiple clusters per object if IoA > {merge_threshold}\n"
    
    print(report)
    
    report_file = model_path / f"segmentation_report_merged_iter_{setup_params.iteration}.txt"
    with open(report_file, "w") as f: f.write(report)
    return final_miou, final_mbiou

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--merge-threshold", type=float, default=0.25, help="IoA threshold to merge a cluster into a GT instance")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--eval-all", action="store_true", help="Evaluate on both train and test cameras")
    args = parser.parse_args()
    evaluate_segmentation(Path(args.model_path), args.iteration, args.threshold, args.device, args.save_masks, args.merge_threshold, args.eval_all)
