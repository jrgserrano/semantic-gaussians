from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import os
import re

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from opensplat3d.gaussian_renderer import render
from opensplat3d.language import LanguageModel
from opensplat3d.language.utils import (
    CropParams,
    RenderParams,
    masks_to_crops,
    seg_pad_resize_masq,
)
from opensplat3d.params import PipeParams
from opensplat3d.utils.setup_utils import SetupParams, setup

from google import genai
from google.genai import types

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-robotics-er-1.5-preview", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"] {"allow-input":true, isTemplate: true}
MODEL_ID = "gemma-3-27b-it"

import time

def call_gemini(prompt: str, images, config=None) -> str:
    default_config = types.GenerateContentConfig(
        temperature=0.5,
    )

    if config is None:
        config = default_config
    
    if not isinstance(images, list):
        images = [images]

    contents = images + [prompt]

    # Added automatic retry wrapper for 429 Rate Limit (Free tier 30 RPM limit)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or attempt < max_retries - 1:
                print(f"  [API Rate Limit Hit] Retrying in {2**(attempt+1)}s...")
                time.sleep(2 ** (attempt + 1))
            else:
                raise e
    return "Unidentified object."

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
    new_w, new_h = width * expansion_ratio, height * expansion_ratio
    
    left = max(0, int(center_x - new_w / 2))
    top = max(0, int(center_y - new_h / 2))
    right = min(image_pil.width, int(center_x + new_w / 2))
    bottom = min(image_pil.height, int(center_y + new_h / 2))
    
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
            'score': (visible_pts / label_count) * (area),
            'view_dir': view_dir,
            'pred_mask': pred_mask.cpu().numpy()
        })
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

def get_vlm_descriptions(instance_id, label, setup_params, pipe_params, bg, bg_color, prompt, cameras, labels, debug=False, debug_dir=None):

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

    raw_descriptions = call_gemini(prompt, images_for_vlm)
    
    return raw_descriptions

@dataclass
class VLMDebugInfo:
    label_id: int
    descriptions: list[str]
    similarity:list[float]
    selected_indices:list[int]

@dataclass
class VLM:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_id = model_id.lower()
        print(f"Loading VLM model: {model_id}...")
        
        if "qwen" in self.model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )

    @torch.no_grad()
    def get_description(self, zoom_crops: torch.Tensor, full_crops: torch.Tensor | None, prompt: str) -> list[str]:
        # zoom_crops, full_crops: (B, C, H, W) uint8 [0, 255]
        descriptions = []

        for i in range(zoom_crops.shape[0]):
            img_zoom = Image.fromarray(zoom_crops[i].permute(1, 2, 0).cpu().numpy())
            img_full = None
            if full_crops is not None:
                img_full = Image.fromarray(full_crops[i].permute(1, 2, 0).cpu().numpy())

            if "qwen" in self.model_id:
                from qwen_vl_utils import process_vision_info
                
                content = [{"type": "image", "image": img_zoom}]
                if img_full is not None:
                    content.append({"type": "image", "image": img_full})
                content.append({"type": "text", "text": prompt})

                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
                
                try:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=60,
                        do_sample=False,
                    )
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    description = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    # Sanitize markdown junk and truncate to ~30-40 CLIP tokens max
                    import re
                    description = re.sub(r'!\[.*?\]\(.*?\)', '', description)  # remove ![]()
                    description = re.sub(r'https?://\S+', '', description)      # remove URLs
                    description = description.strip()[:100]
                    if not description:
                        description = "Unidentified object or visual noise."
                except RuntimeError as e:
                    print(f"\n!!! VLM generation error for image {i}: {e}. Using fallback description.")
                    description = "Unidentified object or visual noise."
            else:
                # Llava 1.5 supports multiple images if provided in a list
                if img_full is not None:
                    inputs = self.processor(text="USER: <image>\n<image>\n" + prompt + " ASSISTANT:", images=[img_zoom, img_full], return_tensors="pt").to("cuda")
                else:
                    inputs = self.processor(text="USER: <image>\n" + prompt + " ASSISTANT:", images=img_zoom, return_tensors="pt").to("cuda")

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                )
                full_text = self.processor.decode(output[0], skip_special_tokens=True)
                description = full_text.split("ASSISTANT:")[-1].strip()
                import re
                description = re.sub(r'!\[.*?\]\(.*?\)', '', description)
                description = re.sub(r'https?://\S+', '', description)
                description = description.strip()[:100]
                if not description:
                    description = "Unidentified object or visual noise."

            descriptions.append(description)

            # Cleanup
            if "inputs" in locals():
                del inputs
            if "output" in locals():
                del output
            if "generated_ids" in locals():
                del generated_ids
            torch.cuda.empty_cache()

        return descriptions

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
        
        distances = 1.0 - similarities

    results = []
    for i, p in enumerate(parsed_respuestas):
        sim = similarities[i].item()
        dist = distances[i].item()
        results.append({'desc': p['desc'], 'id': p['id'], 'sim': sim, 'dist': dist})

        
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
                f.write(f"{r['desc']}: {r['sim']}\n")

    best_option = min(results, key=lambda x: x['dist'])

    if debug:
        with open(label_debug_dir / "best_description.txt", "w") as f:
            f.write(f"{best_option['desc']} ({best_option['id']}): {best_option['sim']}\n")

    return best_option['desc'], best_option['sim'], best_option['id']


@torch.no_grad()
def compute_descriptions(
    setup_params: SetupParams,
    lang_model: LanguageModel,
    crop_params: CropParams,
    use_rendering: bool,
    pred_threshold: float,
    topk: int,
    vlm_model_id: str,
    debug: bool = False,
    debug_dir: Path | None = None,
):
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

    # vlm = VLM(vlm_model_id) # Disabled to use API only

    prompt = """
    You will receive 3 images of the exact same target object captured from 3 different viewpoints.
    
    The target object is highlighted with a bright RED OUTLINE in each image. To help you focus, the background outside the red outline is slightly darkened.
    Analyze the 3 images carefully to understand the object's 3D shape and details. 
    
    Identify and describe the object inside the red outline. Focus only on the object's attributes: 
    - Name or category of the object.
    - Color, texture, and materials.
    - Shape and dimensions.
    - Any text, labels, or patterns written on it.
    - Probable usage.

    Do NOT mention the room, the scene, or the environment. No conversing, no markdown formatting.
    
    You must provide 3 possible candidate descriptions (hypotheses), each representing a valid interpretation of what the object could be and an identifier (one word) for each description.
    
    Output format: 
    1. [First concise description]:[identifier]
    2. [Second concise description]:[identifier]
    3. [Third concise description]:[identifier]
    """

    results = {}

    # top 20 largest clusters to speed up processing
    label_sizes = []
    for idx, label in enumerate(unique_labels):
        if valid_mask[idx]:
            label_sizes.append((idx, (labels == label).sum().item()))
            
    # sort by size descending and keep top 20
    label_sizes.sort(key=lambda x: x[1], reverse=True)
    top_20_indices = {x[0] for x in label_sizes[:20]}
    
    for idx in range(len(valid_mask)):
        if idx not in top_20_indices:
            valid_mask[idx] = False
            
    # unique_labels were computed from labels.unique() which are sorted
    for idx, label in enumerate(tqdm(unique_labels, desc="Generating VLM descriptions")):
        if not valid_mask[idx]:
            continue

        vlm_raw_descriptions = get_vlm_descriptions(idx, label, setup_params, pipe_params, bg, bg_color, prompt, cameras, labels, debug, debug_dir)

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
    parser.add_argument("--levels", type=int, default=0, help="Number of crop levels (0 for whole object)")
    parser.add_argument("--rendering", action="store_true", help="Use rendering for crops")
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM model ID")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save images and info")
    parser.add_argument("--debug-dir", type=Path, help="Directory to save debug images")

    args = parser.parse_args()

    setup_params = setup(args.model_dir)
    lang_model = LanguageModel(args.lang_model)
    
    # Default params for crops for VLM
    crop_params = CropParams(
        lang_model.img_size,
        args.levels,
        True, # masked
        1.5, # expansion ratio
        False, # dynamic ratio
        0.0, # alpha blend
    )

    compute_descriptions(
        setup_params,
        lang_model,
        crop_params,
        args.rendering,
        0.2, # pred_threshold
        args.topk,
        args.vlm,
        args.debug,
        args.debug_dir,
    )
