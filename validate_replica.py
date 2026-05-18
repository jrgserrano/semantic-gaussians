import json
import numpy as np
import os

def iou_3d(box1_center, box1_size, box2_center, box2_size):
    """Calcula el IoU (Intersection over Union) de dos bounding boxes 3D."""
    min1 = np.array(box1_center) - np.array(box1_size) / 2
    max1 = np.array(box1_center) + np.array(box1_size) / 2
    
    min2 = np.array(box2_center) - np.array(box2_size) / 2
    max2 = np.array(box2_center) + np.array(box2_size) / 2
    
    min_inter = np.maximum(min1, min2)
    max_inter = np.minimum(max1, max2)
    
    if np.any(min_inter >= max_inter):
        return 0.0
    
    vol_inter = np.prod(max_inter - min_inter)
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    
    return vol_inter / (vol1 + vol2 - vol_inter)

def main():
    instances_path = "/home/ubuntu/semantic-gaussians/outputs/Replica/room0/20260427225701-2d8f5fdd/instances.json"
    info_path = "/home/ubuntu/datasets/replica_v1/room_0/habitat/info_semantic.json"
    
    print(f"Cargando predicciones: {instances_path}")
    with open(instances_path, 'r') as f:
        instances_data = json.load(f)
        
    print(f"Cargando ground-truth: {info_path}")
    with open(info_path, 'r') as f:
        info_data = json.load(f)
        
    info_objects = info_data['objects']
    
    # Extraer el ID sin el número como indica el usuario (ej: 'card_0' -> 'card')
    # Y cruzarlo con el ground truth usando el IoU de las bounding boxes 3D.
    
    iou_threshold = 0.1
    matched_instances = 0
    total_predictions = len(instances_data['instances'])
    
    validation_results = []

    for inst_key, inst_val in instances_data['instances'].items():
        # Obtener el ID sin número
        predicted_class = inst_key.rsplit('_', 1)[0]
        
        pred_center = inst_val['bbox']['center']
        pred_size = inst_val['bbox']['size']
        
        best_iou = 0
        best_gt_obj = None
        
        # Buscar el objeto de ground truth con mayor superposición (IoU)
        for gt_obj in info_objects:
            gt_center = gt_obj['oriented_bbox']['abb']['center']
            gt_size = gt_obj['oriented_bbox']['abb']['sizes']
            
            iou = iou_3d(pred_center, pred_size, gt_center, gt_size)
            if iou > best_iou:
                best_iou = iou
                best_gt_obj = gt_obj
                
        is_match = False
        gt_class_name = None
        
        if best_iou > iou_threshold and best_gt_obj is not None:
            gt_class_name = best_gt_obj['class_name']
            # Validación simple por nombre exacto
            if predicted_class.lower() == gt_class_name.lower():
                is_match = True
                matched_instances += 1
                
        validation_results.append({
            "instance_key": inst_key,
            "predicted_id_without_number": predicted_class,
            "best_iou": best_iou,
            "matched_gt_class": gt_class_name,
            "is_exact_match": is_match,
            "raw_results_dict": inst_val.get('results')
        })
        
    print("\n--- Resultados de la Validación ---")
    print(f"Total de instancias predichas: {total_predictions}")
    print(f"Coincidencias exactas (IoU > {iou_threshold} y nombre igual): {matched_instances}")
    
    output_validation = "/home/ubuntu/semantic-gaussians/validation_results.json"
    with open(output_validation, 'w') as f:
        json.dump(validation_results, f, indent=4)
        
    print(f"\nResultados detallados guardados en: {output_validation}")
    print("Nota: Las predicciones de vocabulario abierto (ej. 'device') pueden no coincidir exactamente con las clases de Replica (ej. 'cabinet').")
    print("Se recomienda utilizar una métrica de similitud semántica (como CLIP embeddings) sobre la llave 'matched_gt_class' en lugar de igualdad de strings exacta.")

if __name__ == "__main__":
    main()
