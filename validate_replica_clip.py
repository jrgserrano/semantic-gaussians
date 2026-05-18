import json
import numpy as np
import torch
import clip
from tqdm import tqdm

def iou_3d(box1_center, box1_size, box2_center, box2_size):
    min1 = np.array(box1_center) - np.array(box1_size) / 2
    max1 = np.array(box1_center) + np.array(box1_size) / 2
    
    min2 = np.array(box2_center) - np.array(box2_size) / 2
    max2 = np.array(box2_center) + np.array(box2_size) / 2
    
    min_inter = np.maximum(min1, min2)
    max_inter = np.minimum(max1, max2)
    
    if np.any(min_inter >= max_inter):
        return 0.0, 0.0, 0.0
    
    vol_inter = np.prod(max_inter - min_inter)
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    
    iou = vol_inter / (vol1 + vol2 - vol_inter)
    pred_coverage = vol_inter / vol1  # Qué porcentaje de mi predicción es el objeto real
    gt_coverage = vol_inter / vol2    # Qué porcentaje del objeto real es mi predicción
    
    return float(iou), float(pred_coverage), float(gt_coverage)

def get_segmentation_status(iou, pred_cov, gt_cov, threshold=0.1):
    if iou == 0.0 and pred_cov == 0.0 and gt_cov == 0.0:
        return "no-match"
    
    # Si mi predicción está casi toda contenida en el GT (pred_cov alto)
    # pero cubre una pequeña parte del GT (gt_cov bajo) -> es un trozo del objeto real
    if pred_cov > 0.4 and gt_cov < 0.3:
        return "over-segmented"
        
    # Si el GT está casi todo contenido en mi predicción (gt_cov alto)
    # pero es una pequeña parte de mi predicción (pred_cov bajo) -> mi obj engulle varios
    if gt_cov > 0.4 and pred_cov < 0.3:
        return "under-segmented"
        
    if iou > threshold:
        return "good-match"
        
    return "partial-match"

def main():
    instances_path = "/home/ubuntu/semantic-gaussians/outputs/Replica/room1/20260508111114-36ef0444/instances.json"
    info_path = "/home/ubuntu/datasets/replica_v1/room_1/habitat/info_semantic.json"
    
    print(f"Cargando predicciones: {instances_path}")
    with open(instances_path, 'r') as f:
        instances_data = json.load(f)
        
    print(f"Cargando ground-truth: {info_path}")
    with open(info_path, 'r') as f:
        info_data = json.load(f)
        
    info_objects = info_data['objects']
    
    print("Cargando modelo CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    print("Calculando embeddings de clases de ground truth...")
    gt_classes = list(set([obj['class_name'] for obj in info_objects]))
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in gt_classes]).to(device)
    
    with torch.no_grad():
        gt_embeddings = model.encode_text(text_inputs)
        gt_embeddings /= gt_embeddings.norm(dim=-1, keepdim=True)
    
    gt_class_to_embedding = {gt_classes[i]: gt_embeddings[i] for i in range(len(gt_classes))}
    
    iou_threshold = 0.1
    similarity_threshold = 0.6
    
    matched_instances_iou_only = 0
    matched_instances_exact = 0
    matched_instances_semantic = 0
    
    status_counts = {"good-match": 0, "over-segmented": 0, "under-segmented": 0, "partial-match": 0, "no-match": 0}
    
    total_predictions = len(instances_data['instances'])
    validation_results = []
    
    print("Evaluando predicciones...")
    for inst_key, inst_val in tqdm(instances_data['instances'].items()):
        predicted_class = inst_key.rsplit('_', 1)[0]
        
        with torch.no_grad():
            pred_text = clip.tokenize(f"a photo of a {predicted_class}").to(device)
            pred_emb = model.encode_text(pred_text)
            pred_emb /= pred_emb.norm(dim=-1, keepdim=True)
            pred_emb = pred_emb.squeeze(0)
            
        pred_center = inst_val['bbox']['center']
        pred_size = inst_val['bbox']['size']
        
        best_iou = 0
        best_pred_cov = 0
        best_gt_cov = 0
        best_gt_obj = None
        
        for gt_obj in info_objects:
            abb = gt_obj.get('oriented_bbox', {}).get('abb', {})
            ori = gt_obj.get('oriented_bbox', {}).get('orientation', {})
            
            gt_center = np.array(abb.get('center', [0,0,0]))
            gt_size = np.array(abb.get('sizes', [0,0,0]))
            
            # Obtener el cuaternión [x, y, z, w]
            rot_gt = ori.get('rotation', [0,0,0,1])
            
            # Matriz de rotación desde el cuaternión (scipy o calculada a mano)
            x, y, z, w = rot_gt
            rot_mat = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
            ])
            
            # Calcular las 8 esquinas locales
            dx, dy, dz = gt_size / 2
            corners_local = np.array([
                [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
                [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz]
            ])
            
            # Transformar a mundo (rotación + traslación)
            corners_world = (rot_mat @ (corners_local + gt_center).T).T
            
            # Obtener el nuevo AABB (ejes alineados con el mundo)
            gt_min = corners_world.min(axis=0)
            gt_max = corners_world.max(axis=0)
            
            gt_center_aabb = (gt_min + gt_max) / 2
            gt_size_aabb = gt_max - gt_min
            
            iou, pred_cov, gt_cov = iou_3d(pred_center, pred_size, gt_center_aabb.tolist(), gt_size_aabb.tolist())
            
            # Buscamos el ground truth con el que tengamos mayor intersección/solapamiento
            # Puedes maximizar iou, pero a veces maximizar la intersección cruda o covs funciona
            # Para evaluación general, nos quedamos con el de mayor IoU
            if iou > best_iou:
                best_iou = iou
                best_pred_cov = pred_cov
                best_gt_cov = gt_cov
                best_gt_obj = gt_obj
                
        is_exact_match = False
        is_semantic_match = False
        semantic_similarity = 0.0
        gt_class_name = None
        gt_id = None
        
        status = get_segmentation_status(best_iou, best_pred_cov, best_gt_cov, threshold=iou_threshold)
        status_counts[status] += 1
        
        if best_gt_obj is not None:
            gt_class_name = best_gt_obj['class_name']
            gt_id = best_gt_obj['id']
            
            if best_iou > iou_threshold:
                matched_instances_iou_only += 1
                
            if predicted_class.lower() == gt_class_name.lower():
                is_exact_match = True
                if best_iou > iou_threshold: matched_instances_exact += 1
                
            gt_emb = gt_class_to_embedding[gt_class_name]
            semantic_similarity = torch.dot(pred_emb, gt_emb).item()
            
            if semantic_similarity > similarity_threshold and best_iou > iou_threshold:
                is_semantic_match = True
                matched_instances_semantic += 1
                
        validation_info = {
            "predicted_id_without_number": predicted_class,
            "best_iou": best_iou,
            "pred_coverage": best_pred_cov,
            "gt_coverage": best_gt_cov,
            "segmentation_status": status,
            "matched_gt_class": gt_class_name,
            "matched_gt_id": gt_id,
            "is_exact_match": is_exact_match,
            "semantic_similarity": semantic_similarity,
            "is_semantic_match": is_semantic_match
        }
                
        validation_results.append({
            "instance_key": inst_key,
            **validation_info,
            "raw_results_dict": inst_val.get('results')
        })
        
        # También actualizamos el instances_data
        instances_data['instances'][inst_key]['validation'] = validation_info
        
    valid_similarities = [res['semantic_similarity'] for res in validation_results if res.get('semantic_similarity') is not None]
    avg_sim = sum(valid_similarities) / len(valid_similarities) if valid_similarities else 0.0
    
    spatial_matches = [res['semantic_similarity'] for res in validation_results if res.get('semantic_similarity') is not None and res.get('best_iou', 0) > 0]
    avg_sim_spatial = sum(spatial_matches) / len(spatial_matches) if spatial_matches else 0.0

    print("\n--- Overall Results ---")
    print(f"Total instances: {total_predictions}")
    print(f"Mean semantic similarity (all instances): {avg_sim:.4f}")


    
    # Comprobar cuántas instancias apuntan al mismo ground truth id (clara sobresegmentación)
    gt_to_preds = {}
    for res in validation_results:
        if res['matched_gt_id'] is not None and res['segmentation_status'] in ['good-match', 'over-segmented']:
            gt_id = res['matched_gt_id']
            if gt_id not in gt_to_preds:
                gt_to_preds[gt_id] = []
            gt_to_preds[gt_id].append(res['instance_key'])
            
    oversegmented_gt = {k: v for k, v in gt_to_preds.items() if len(v) > 1}
    
    output_validation = "/home/ubuntu/semantic-gaussians/validation_results_clip.json"
    with open(output_validation, 'w') as f:
        json.dump(validation_results, f, indent=4)
        
    with open(instances_path, 'w') as f:
        json.dump(instances_data, f, indent=4)
        
    print(f"\nResultados detallados guardados en:\n- {output_validation}\n- {instances_path}")

if __name__ == "__main__":
    main()
