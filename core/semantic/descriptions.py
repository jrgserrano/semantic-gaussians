from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import os
import re
import torch.backends.cudnn
torch.backends.cudnn.enabled = False

from core.gaussian_renderer import render
from core.language import LanguageModel
from core.language.utils import (
    CropParams,
    RenderParams,
    masks_to_crops,
    seg_pad_resize_masq,
)
from core.params import PipeParams
from core.utils.setup_utils import SetupParams, setup

def load_embeddings(model_path: Path, lang_model_type: str) -> dict:
    output_file = model_path / f"{lang_model_type}_embeddings.pth"
    if not output_file.exists():
        print(f"Embeddings file not found: {output_file}")
        return None
    print(f"Loading embeddings from: {output_file}")
    return torch.load(output_file)

def get_dynamic_expansion(mask_np):
    h, w = mask_np.shape
    area_ratio = np.sum(mask_np) / (h * w)
    
    if area_ratio < 0.005:
        return 3.5
    elif area_ratio > 0.2:
        return 1.1
    else:
        t = (area_ratio - 0.005) / (0.2 - 0.005)
        return 3.5 + t * (1.1 - 3.5)

def crop_to_object(image_pil, mask_np, expansion_ratio=1.5):
    if not np.any(mask_np):
        return image_pil
    
    coords = np.argwhere(mask_np)
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    
    width = xmax - xmin
    height = ymax - ymin
    
    center_x, center_y = xmin + width / 2, ymin + height / 2
    new_w = max(10, width * expansion_ratio)
    new_h = max(10, height * expansion_ratio)
    
    left = max(0, int(center_x - new_w / 2))
    top = max(0, int(center_y - new_h / 2))
    right = min(image_pil.width, int(center_x + new_w / 2))
    bottom = min(image_pil.height, int(center_y + new_h / 2))
    
    if right <= left + 1 or bottom <= top + 1:
        return image_pil
        
    return image_pil.crop((left, top, right, bottom))

def prepare_vlm_image(image: torch.Tensor, mask: torch.Tensor, darken=True, zoom=True):
    mask_float = mask.float()
    mask_np = mask.cpu().numpy()
    
    bg_image = image * 0.3 if darken else image
    vis_image = image * mask_float + bg_image * (1 - mask_float)
    
    kernel = torch.ones(3, 3, device=image.device)
    dilated_mask = (F.conv2d(mask_float.unsqueeze(0).unsqueeze(0), 
                             kernel.unsqueeze(0).unsqueeze(0), padding=1) > 0).squeeze()
    outline = dilated_mask & ~mask
    if vis_image.shape[0] == 3: vis_image[0][outline] = 1.0
    
    img_np = (vis_image.permute(1, 2, 0).cpu().detach().numpy() * 255).astype('uint8')
    pil_img = Image.fromarray(img_np)
    
    if zoom:
        expansion = get_dynamic_expansion(mask_np)
        pil_img = crop_to_object(pil_img, mask_np, expansion_ratio=expansion)
        
    return pil_img

@dataclass
class Stats:
    cam_idx: int
    pred_mask: np.ndarray
    area: int
    label_count: int
    visible_count: int

def find_best_views(instance_id, label_id, setup_params, pipe_params, bg, cameras, n_views=3, pred_threshold=0.2, labels=None):
    """
    Busca N vistas de alta calidad que sean lo más diversas posible geométricamente.
    """

    device = setup_params.device
    label_mask = (labels == int(instance_id))
    label_count = int(label_mask.sum().item())
    
    if label_count == 0: return []

    all_stats = []
    mask_color = torch.zeros(setup_params.gaussians.get_xyz.shape[0], 3, device=device)
    mask_color[label_mask] = 1.0

    for i, cam in enumerate(cameras):
        render_pkg = render(cam, setup_params.gaussians, pipe_params, bg, 
                            setup_params.model_params.sh_degree, override_color=mask_color)
        
        image = render_pkg.render.clamp(0, 1).permute(1, 2, 0)
        pred_mask = image.mean(dim=-1) > pred_threshold
        area = int(pred_mask.sum().item())
        visible_pts = int((render_pkg.visibility_filter.cpu() & label_mask.cpu()).sum().item())
        
        v2w = cam.world_view_transform.inverse()
        view_dir = v2w[2, :3].detach().clone() 
        
        all_stats.append({
            'cam_idx': i,
            'area': area,
            'visible_pts': visible_pts,
            'view_dir': view_dir,
            'pred_mask': pred_mask.cpu().numpy()
        })
        
    max_area = max([s['area'] for s in all_stats]) if all_stats else 0
    if max_area > 0:
        for s in all_stats:
            s['score'] = (s['visible_pts'] / label_count) * (s['area'] / max_area)
    else:
        for s in all_stats: s['score'] = 0
    candidates = [s for s in all_stats if s['score'] > 0]
    if not candidates: return []

    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    selected = [candidates[0]]
    
    pool = candidates[:len(candidates)//4]
    if len(pool) < n_views: pool = candidates

    for _ in range(n_views - 1):
        best_diverse_cand = None
        min_max_sim = 1.0

        for cand in pool:
            if any(c['cam_idx'] == cand['cam_idx'] for c in selected): continue
            
            max_sim = 0
            for s in selected:
                sim = F.cosine_similarity(cand['view_dir'].unsqueeze(0), s['view_dir'].unsqueeze(0)).item()
                max_sim = max(max_sim, sim)
            
            if max_sim < min_max_sim:
                min_max_sim = max_sim
                best_diverse_cand = cand
        
        if best_diverse_cand:
            selected.append(best_diverse_cand)

    return selected

def get_vlm_descriptions(instance_id, label, setup_params, pipe_params, bg, prompt, cameras, labels, debug=False, debug_dir=None, vlm_model=None):

    label_id = int(label.item())

    diverse_views = find_best_views(instance_id, label_id, setup_params, pipe_params, bg, cameras, n_views=3, labels=labels)
    if not diverse_views: return

    images_for_vlm = []
    
    for view in diverse_views:
        cam = cameras[view['cam_idx']]
        real_render = render(cam, setup_params.gaussians, pipe_params, bg, 
                             setup_params.model_params.sh_degree).render
        mask = torch.from_numpy(view['pred_mask']).to(setup_params.device)
        vlm_img = prepare_vlm_image(real_render, mask)

        if debug:
            label_debug_dir = debug_dir / str(label_id)
            label_debug_dir.mkdir(parents=True, exist_ok=True)
            vlm_img.save(label_debug_dir / f"view_{view['cam_idx']}.png")
                
        images_for_vlm.append(vlm_img)

    if vlm_model is not None:
        raw_descriptions = vlm_model.get_description_from_pil(images_for_vlm, prompt)
    else:
        raw_descriptions = "Unidentified object."
    
    return raw_descriptions

@dataclass
class VLMDebugInfo:
    label_id: int
    descriptions: list[str]
    similarity:list[float]
    selected_indices:list[int]

class OllamaVLM:
    def __init__(self, model_id: str):
        self.model_id = model_id.replace("ollama/", "").replace("google/", "")
        print(f"Loading VLM via Ollama: {self.model_id}...")

    def get_description_from_pil(self, images: list, prompt: str) -> str:
        import ollama
        from io import BytesIO

        img_bytes = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_bytes.append(buffered.getvalue())
        
        try:
            response = ollama.chat(
                model=self.model_id,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': img_bytes,
                    },
                ],
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Failed to call Ollama via library: {e}")
            return "Error calling Ollama."



def calculate_vlm_similarity(instance_id, descriptions, setup_params, lang_model):
    model_path = setup_params.model_path
    device = setup_params.device
    
    emb_path = model_path / f"{lang_model.model_type}_embeddings.pth"
    if not emb_path.exists():
        print(f"Error: No se encontró el archivo de embeddings en {emb_path}")
        return None
    
    emb_data = torch.load(emb_path, map_location=device)
    all_inst_embeddings = torch.from_numpy(emb_data["embeddings"]).to(device)
    
    labels = torch.from_numpy(np.load(model_path / "clustering" / "labels.npy"))
    unique_labels = labels.unique()
    unique_labels = unique_labels[unique_labels != -1]
    
    try:
        inst_idx = (unique_labels == int(instance_id)).nonzero(as_tuple=True)[0].item()
        instance_embedding = all_inst_embeddings[inst_idx]
    except (ValueError, IndexError):
        print(f"Error: No hay un embedding guardado para la instancia {instance_id}")
        return None

    raw_respuestas = [line.strip() for line in re.findall(r'\d\.\s*(.*)', descriptions)]

    if not raw_respuestas:
        raw_respuestas = [l.strip() for l in descriptions.split('\n') if len(l.strip()) > 5]
    
    if not raw_respuestas:
        raw_respuestas = [descriptions]

    parsed_respuestas = []
    for r in raw_respuestas:
        if ":" in r:
            desc_part, id_part = r.rsplit(":", 1)
            parsed_respuestas.append({"desc": desc_part.strip(), "id": id_part.strip()})
        else:
            parsed_respuestas.append({"desc": r.strip(), "id": "unknown"})

    formatted_texts = [lang_model.prompt_template.format(p["desc"]) for p in parsed_respuestas]

    
    with torch.no_grad():
        text_embeddings = lang_model.embed_text(formatted_texts, normalize=True)
        text_embeddings = text_embeddings.to(device)
        
        instance_embedding = instance_embedding / instance_embedding.norm(p=2, dim=-1, keepdim=True)
        
        similarities = (text_embeddings @ instance_embedding.unsqueeze(1)).squeeze(1)

    results = []
    for i, p in enumerate(parsed_respuestas):
        sim = similarities[i].item()
        results.append({'desc': p['desc'], 'id': p['id'], 'sim': sim})

        
    return results


def get_best_vlm_description(instance_id, descriptions, setup_params, lang_model, debug=False, debug_dir=None):
    results = calculate_vlm_similarity(instance_id, descriptions, setup_params, lang_model)
    
    if not results:
        return "No se pudo determinar la mejor descripción."

    if debug:
        label_debug_dir = debug_dir / str(instance_id)
        label_debug_dir.mkdir(parents=True, exist_ok=True)
        with open(label_debug_dir / "results.txt", "w") as f:
            for r in results:
                f.write(f"{r['desc']} : {r['id']} : {r['sim']}\n")

    best_option = max(results, key=lambda x: x['sim'])

    if debug:
        with open(label_debug_dir / "best_description.txt", "w") as f:
            f.write(f"{best_option['desc']} : {best_option['id']} : {best_option['sim']}\n")

    return best_option['desc'], best_option['sim'], best_option['id']


@torch.no_grad()
def compute_descriptions(
    setup_params: SetupParams,
    lang_model: LanguageModel,
    topk: int,
    vlm_model_id: str,
    debug: bool = False,
    debug_dir: Path | None = None,
):
    torch.cuda.set_device(0)
    model_path = setup_params.model_path
    if debug and debug_dir is None:
        debug_dir = model_path / "vlm_debug"
    
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode enabled. Saving images and info to: {debug_dir}")
    
    # Load embeddings generated by embed.py
    embeddings_data = load_embeddings(model_path, lang_model.model_type)
    if embeddings_data is None:
        print("Required embeddings not found. Please run embed.py first.")
        return

    embeddings = torch.from_numpy(embeddings_data["embeddings"])
    valid_mask = embeddings_data["valid"]
    
    # CRITICAL: Fix for illegal memory access during rasterization
    # Since Qwen loaded with device_map="auto", the active CUDA context could be 0, 
    # but the Gaussians are loaded based on setup_params.device.
    if setup_params.device.type == 'cuda':
        torch.cuda.set_device(setup_params.device)
    
    labels = torch.from_numpy(np.load(model_path / "clustering" / "labels.npy"))
    unique_labels = labels.unique()
    unique_labels = unique_labels[unique_labels != -1]
    
    cameras = setup_params.scene.get_train_cameras()
    pipe_params = PipeParams()
    bg_color = [1, 1, 1] if setup_params.model_params.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device=setup_params.device)
    
    render_params = RenderParams(
        setup_params.gaussians,
        setup_params.model_params,
        cameras,
        pipe_params,
        bg,
    )

    vlm = OllamaVLM(vlm_model_id)

    prompt = """
    You will receive 3 images of the exact same target object captured from 3 different viewpoints.
    
    The target object is highlighted with a bright RED OUTLINE in each image.
    The background outside the red outline is intentionally darkened to direct your attention to the object.
    Carefully analyze all 3 views to understand the object's 3D shape, size, and details.
    
    Describe only the object inside the red outline. Completely ignore the environment, room, and other elements.
    Focus only on the object's attributes: 
    - Category or type of object (be specific, not generic)
    - Dominant color and secondary colors
    - Observable material and texture
    - Overall shape and estimated proportions
    - Any visible markings, labels, numbers, or patterns on the surface
    - Probable function or usage

    Generating hypotheses:
    Provide 3 different interpretations of the object. Each hypothesis must:
    - Be a CONCISE description (1-2 sentences maximum)
    - Represent a valid but different categorization from the others
    - Include a SINGLE-WORD identifier that is the most descriptive name for the object

    Output restrictions:
    - No additional explanations, no conversation, no markdown formatting.
    - Only the required format.
    - Be direct and concise.

    Output format: 
    1. [First concise description]:[identifier]
    2. [Second concise description]:[identifier]
    3. [Third concise description]:[identifier]
    """

    results = {}

    label_sizes = []
    for idx, label in enumerate(unique_labels):
        if valid_mask[idx]:
            label_sizes.append((idx, (labels == label).sum().item()))
            
    # unique_labels were computed from labels.unique() which are sorted
    for idx, label in enumerate(tqdm(unique_labels, desc="Generating VLM descriptions")):
        if not valid_mask[idx]:
            continue

        vlm_raw_descriptions = get_vlm_descriptions(idx, label, setup_params, pipe_params, bg, prompt, cameras, labels, debug, debug_dir, vlm)

        if not vlm_raw_descriptions:
            print(f"No VLM descriptions generated for instance {idx}")
            continue

        best_description, best_sim, best_id = get_best_vlm_description(idx, vlm_raw_descriptions, setup_params, lang_model, debug, debug_dir)

        label_id = int(label.item())

        results[label_id] = {
            "description": best_description,
            "similarity": best_sim,
            "identifier": best_id
        }

        
        torch.cuda.empty_cache()
        
        # Iterative Auto-Save so we don't lose progress if interrupted
        output_path = model_path / f"descriptions.pth"
        torch.save(results, output_path)

    print(f"\nDescriptions saved to: {output_path}")

    print("\nGenerating Bounding Boxes...")
    from core.semantic.generate_bboxes import generate_bboxes
    from core.semantic.export_json import export_instances_json
    
    generate_bboxes(model_path)
    print("\nExporting Instances JSON...")
    export_instances_json(model_path, lang_model.model_type)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="Path to the model directory")
    parser.add_argument(
        "--lang-model",
        type=str,
        choices=["clip", "siglip", "masqclip"],
        default="masqclip",
        help="Language model",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of best views to describe")
    parser.add_argument("--vlm", type=str, default="ollama/gemma4:31b", help="VLM model ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save images and info")
    parser.add_argument("--debug-dir", type=Path, help="Directory to save debug images")

    args = parser.parse_args()

    setup_params = setup(args.model_dir)
    lang_model = LanguageModel(args.lang_model)

    compute_descriptions(
        setup_params,
        lang_model,
        args.topk,
        args.vlm,
        args.debug,
        args.debug_dir,
    )
