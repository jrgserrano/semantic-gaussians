import argparse
import json
from pathlib import Path
import numpy as np
import torch

def export_instances_json(model_path: Path, lang_model_type: str = "masqclip"):
    bbox_path = model_path / "clustering" / "bboxes.pth"
    if not bbox_path.exists():
        print(f"Error: Bounding boxes not found at {bbox_path}")
        return
    bboxes = torch.load(bbox_path, weights_only=False)
    
    labels_path = model_path / "clustering" / "labels.npy"
    if not labels_path.exists():
        print(f"Error: Labels not found at {labels_path}")
        return
    labels = np.load(labels_path)
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
    
    desc_path = model_path / f"descriptions.pth"
    descriptions = {}
    if desc_path.exists():
        desc_data = torch.load(desc_path, weights_only=False)
        for label_id, data in desc_data.items():
            descriptions[label_id] = {
                "label": data["description"],
                "similarity": data["similarity"],
                "identifier": data.get("identifier", "unknown")
            }

    else:
        print(f"Warning: Descriptions not found at {desc_path}. 'results' will use 'unknown'.")

    output_data = {"instances": {}}
    
    for i, label_id in enumerate(sorted(bboxes.keys())):
        if label_id in descriptions:
            raw_id = descriptions[label_id]["identifier"]
            clean_id = "".join(e for e in raw_id if e.isalnum()).lower()
            obj_id = f"{clean_id}_{label_id}"
        else:
            obj_id = f"obj{i}"

        bbox_info = bboxes[label_id]
        
        n_obs = label_counts.get(label_id, 0)
        
        caption = "unknown"
        similarity = 0.0
        if label_id in descriptions:
            desc_info = descriptions[label_id]
            caption = desc_info["label"]
            similarity = desc_info["similarity"]
            
        output_data["instances"][obj_id] = {
            "bbox": {
                "center": bbox_info["center"],
                "size": bbox_info["size"]
            },
            "n_observations": n_obs,
            "results": {
                "caption": caption,
                "similarity": similarity
            }
        }
    
    output_path = model_path /"instances.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Exported {len(output_data['instances'])} instances to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export clusters to JSON format")
    parser.add_argument("model_dir", type=Path, help="Path to the model directory")
    parser.add_argument("--lang-model", type=str, default="masqclip", help="Language model used for descriptions")
    args = parser.parse_args()
    
    export_instances_json(args.model_dir, args.lang_model)
