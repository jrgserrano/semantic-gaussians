import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

dense_json = Path("outputs/Replica/office4_dense_train/20260613170349-f39d30b7/training_stats.json")
colmap_json = Path("outputs/Replica/office4_colmap_train/20260613124546-14a9d92e/training_stats.json")

runs = {
    "COLMAP Init": colmap_json,
    "Dense Init": dense_json
}

metrics_to_plot = ["psnr_train", "ssim_train", "lpips_train"]
titles = ["PSNR (Train)", "SSIM (Train)", "LPIPS (Train)"]

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Comparison: Dense vs COLMAP Initialization (Replica office4)", fontsize=16)

for run_name, json_path in runs.items():
    if not json_path.exists():
        print(f"Missing file: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        stats = json.load(f)
        
    for i, metric in enumerate(metrics_to_plot):
        if metric in stats and len(stats[metric]) > 0:
            iters = [x["iter"] for x in stats[metric]]
            values = [x["value"] for x in stats[metric]]
            axes[i].plot(iters, values, label=run_name, marker='o', markersize=3)

for i, metric in enumerate(metrics_to_plot):
    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(titles[i].split(" ")[0])
    axes[i].legend()
    axes[i].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
out_path = OUTPUT_DIR / "comparison_dense_vs_colmap.png"
plt.savefig(out_path, dpi=150)
print(f"Saved plot: {out_path}")
plt.close()
