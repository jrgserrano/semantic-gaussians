import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

SCENES = ["office1", "office4", "room0", "room1"]
EXPERIMENTS = {
    "Exp1_Initialization": ["exp1_random", "exp1_points"],
    "Exp2_AdaptivePruning": ["exp2_prune_false", "exp2_prune_true"],
    "Exp2_AdaptivePruning_Rand": ["exp2_prune_false_rand", "exp2_prune_true_rand"],
    "Exp3_Depth": ["exp3_depth_0.0", "exp3_depth_1.0"],
    "Exp3_Depth_Rand": ["exp3_depth_0.0_rand", "exp3_depth_1.0_rand"],
    "Exp4_FrameSelection": ["exp4_frames_uniform", "exp4_frames_random"],
    "Exp5_CameraSampler": ["exp5_sampler_fisher", "exp5_sampler_random"]
}
BASE_METRICS = ["psnr", "ssim", "lpips"]

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = Path("outputs/experiments")

for scene in SCENES:
    for exp_name, exp_variants in EXPERIMENTS.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Scene: {scene} | Experiment: {exp_name}", fontsize=16)
        
        for variant in exp_variants:
            variant_dir = EXPERIMENTS_DIR / scene / variant
            stats_files = list(variant_dir.glob("*/training_stats.json"))
            if not stats_files:
                print(f"File not found: {variant_dir}/*/training_stats.json")
                continue
            
            stats_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            stats_file = stats_files[0]
                
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            for i, base_metric in enumerate(BASE_METRICS):
                metric_test = f"{base_metric}_test"
                metric_train = f"{base_metric}_train"
                
                metric_to_plot = None
                label_suffix = ""
                
                if metric_test in stats and len(stats[metric_test]) > 0:
                    metric_to_plot = metric_test
                    label_suffix = "(test)"
                elif metric_train in stats and len(stats[metric_train]) > 0:
                    metric_to_plot = metric_train
                    label_suffix = "(train)"
                    
                if metric_to_plot is not None:
                    iters = [x["iter"] for x in stats[metric_to_plot]]
                    values = [x["value"] for x in stats[metric_to_plot]]
                    # Extract the meaningful part of the variant name
                    label = variant.split('_', 1)[1] if '_' in variant else variant
                    label = f"{label} {label_suffix}"
                    axes[i].plot(iters, values, label=label, marker='o', markersize=3)
                    
        for i, base_metric in enumerate(BASE_METRICS):
            axes[i].set_title(base_metric.upper())
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel(base_metric.upper())
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{scene}_{exp_name}.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
        plt.close()

print(f"All plots generated in {OUTPUT_DIR}")
