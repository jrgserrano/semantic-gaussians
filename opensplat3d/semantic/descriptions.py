from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import os

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

def call_gemini_robotics(prompt: str, images, config=None) -> str:
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

@dataclass
class Stats:
    cam_idx: int
    pred_mask: np.ndarray
    area: int
    label_count: int
    visible_count: int

@dataclass
class VLMDebugInfo:
    label_id: int
    descriptions: list[str]
    similarity:list[float]
    selected_indices:list[int]

import torchvision.transforms.functional as TF

def apply_vlm_visualization(image: torch.Tensor, mask: torch.Tensor, darken: bool = True, red_outline: bool = True):
    # image: (C, H, W) float [0, 1]
    # mask: (H, W) bool
    
    # 1. Darken background
    mask_float = mask.float()
    if darken:
        bg_image = image * 0.3
    else:
        bg_image = image
        
    # 2. Combine: object Original, background Darkened
    vis_image = image * mask_float + bg_image * (1 - mask_float)
    
    # 3. Red outline
    if red_outline:
        import torch.nn.functional as F
        kernel = torch.ones(3, 3, device=image.device)
        dilated_mask = (F.conv2d(mask_float.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1) > 0).squeeze()
        # the mask shape might be smaller due to conv2d, wait actually padding=1 ensures same shape
        # let's be careful with bounds
        outline = dilated_mask & ~mask
        
        vis_image[0][outline] = 1.0 # Red
        vis_image[1][outline] = 0.0
        vis_image[2][outline] = 0.0
        
    return vis_image

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
        

            


def compute_ious(gt_mask: torch.Tensor, pred_mask: torch.Tensor):
    ious = []
    gt_labels = torch.unique(gt_mask)
    gt_labels = gt_labels[gt_labels != -1]
    for label in gt_labels:
        intersection = (gt_mask == label) & pred_mask
        union = (gt_mask == label) | pred_mask
        iou = intersection.sum() / union.sum()
        ious.append(iou)
    return ious


@torch.no_grad()
def compute_view_stats(
    render_params: RenderParams,
    mask_color: torch.Tensor,
    pred_threshold: float,
    label_mask: torch.Tensor,
    label_count: int,
):
    stats: list[Stats] = []
    for i, cam in enumerate(render_params.cameras):
        render_pkg = render(
            cam,
            render_params.gaussians,
            render_params.pipe_params,
            render_params.bg,
            render_params.model_params.sh_degree,
            override_color=mask_color,
            render_features=False,
        )
        assert render_pkg.render is not None, "Rendered image is None"
        image = render_pkg.render.clamp(0, 1).permute(1, 2, 0).contiguous()
        pred_mask = image.mean(dim=-1) > pred_threshold
        area = pred_mask.flatten().sum()

        visible_count = (render_pkg.visibility_filter.cpu() & label_mask).sum()

        stats.append(
            Stats(
                i,
                pred_mask.cpu().numpy(),
                int(area.cpu().item()),
                label_count,
                int(visible_count.item()),
            )
        )
    return stats


@torch.no_grad()
def preprocess_crops(
    render_params: RenderParams,
    stats: list[Stats],
    lang_model: LanguageModel | None,
    crop_params: CropParams,
    use_rendering: bool,
):
    all_crops: dict[str, tuple[list[torch.Tensor], list[torch.Tensor] | None]] = {}
    for stat in stats:
        pred_mask = torch.from_numpy(stat.pred_mask)
        if use_rendering:
            render_pkg = render(
                render_params.cameras[stat.cam_idx],
                render_params.gaussians,
                render_params.pipe_params,
                render_params.bg,
                render_params.model_params.sh_degree,
                render_features=False,
            )
            assert render_pkg.render is not None, "Rendered image is None"
            image = render_pkg.render.clamp(0, 1).mul(255).to(dtype=torch.uint8).cpu()
        else:
            image = (
                render_params.cameras[stat.cam_idx]
                .original_image.clamp(0, 1)
                .mul(255)
                .to(dtype=torch.uint8)
            )

        old_expansion_ration = crop_params.expansion_ratio
        area_ratio = stat.area / (image.shape[1] * image.shape[2])
        if crop_params.dynamic_ratio:
            expansion_ratio = (
                crop_params.expansion_ratio if area_ratio < 0.0075 else 0.1
            )
        else:
            expansion_ratio = crop_params.expansion_ratio

        crop_params.expansion_ratio = expansion_ratio

        crops: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        if crop_params.levels == 0:
            if lang_model is not None:
                assert lang_model.model_type == "masqclip", (
                    "Non-crop mode only available for MasQClip"
                )
            crops_def, crop_masks_def = seg_pad_resize_masq(
                image,
                pred_mask.unsqueeze(0),
                lang_model.img_size if lang_model is not None else crop_params.img_size,
            )
            crops["default"] = (
                lang_model.preprocess_images(
                    crops_def.permute(0, 2, 3, 1).contiguous().numpy()
                ) if lang_model is not None else crops_def,
                crop_masks_def,
            )
        else:
            crops_def, crop_masks_def, _, _ = masks_to_crops(
                image,
                pred_mask.unsqueeze(0),
                crop_params,
                lang_model,
            )
            crops_def = crops_def.squeeze(0)
            crop_masks_def = crop_masks_def.squeeze(0).unsqueeze(1)
            crops["default"] = (
                crops_def,
                crop_masks_def if (lang_model is not None and lang_model.model_type == "masqclip") else (crop_masks_def if lang_model is None else None),
            )

        crop_params.expansion_ratio = old_expansion_ration

        for name, crops_ in crops.items():
            if name not in all_crops:
                all_crops[name] = ([], [])
            all_crops[name][0].append(crops_[0])
            if crops_[1] is not None:
                x = all_crops[name][1]
                if x is not None:
                    x.append(crops_[1])

        for name, crops_ in all_crops.items():
            if crops_[1] is not None and len(crops_[1]) == 0:
                all_crops[name] = (crops_[0], None)

    return {
        name: (
            torch.cat(crops_[0]),
            torch.cat(crops_[1]) if crops_[1] is not None else None,
        )
        for name, crops_ in all_crops.items()
    }

def load_embeddings(model_path: Path, lang_model_type: str) -> dict:
    output_file = model_path / f"{lang_model_type}_embeddings.pth"
    if not output_file.exists():
        print(f"Embeddings file not found: {output_file}")
        return None
    print(f"Loading embeddings from: {output_file}")
    return torch.load(output_file)

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
    # Base prompt (without <image> tags, handled by VLM class)
    # The output must contain the name that best describes the object and its attributes (color, texture, shape, usage, etc.),
    
    prompt = """
    You will receive 3 images of the exact same target object at different zoom levels:
    1) A full-scene context shot.
    2) A medium distance shot.
    3) An extreme close-up of the object.
    
    The target object is highlighted with a bright RED OUTLINE. To further help you focus, the rest of the room (outside the red outline) is slightly darkened.
    Analyze the 3 images carefully. Identify and precisely describe the object inside the red outline.
    
    The output must contain ONLY the name that best describes the object, its attributes (color, texture, shape, text written on it, usage, etc.), and the room it is in. No conversing, no markdown formatting.
    The description must make sense in the context of the room. The output format MUST be a single concise phrase.
    """

    results = {}
    
    # Global collection for best descriptions
    all_best_descriptions = []

    # Filter for top 100 largest clusters to speed up processing
    label_sizes = []
    for idx, label in enumerate(unique_labels):
        if valid_mask[idx]:
            label_sizes.append((idx, (labels == label).sum().item()))
            
    # Sort by size descending and keep top 100
    label_sizes.sort(key=lambda x: x[1], reverse=True)
    top_100_indices = {x[0] for x in label_sizes[:20]}
    
    for idx in range(len(valid_mask)):
        if idx not in top_100_indices:
            valid_mask[idx] = False
            
    # unique_labels were computed from labels.unique() which are sorted
    for idx, label in enumerate(tqdm(unique_labels, desc="Generating VLM descriptions")):
        if not valid_mask[idx]:
            continue
            
        label_id = int(label.item())
        label_mask = labels == label
        mask_color = torch.zeros(setup_params.gaussians.get_xyz.shape[0], 3, device=setup_params.device)
        mask_color[label_mask] = 1.0 # White mask

        stats = compute_view_stats(
            render_params,
            mask_color,
            pred_threshold,
            label_mask,
            int(label_mask.sum().item()),
        )
        
        if not stats:
            continue
            
        max_area = max([x.area for x in stats])
        if max_area == 0:
            continue

        selected_stats = sorted(
            stats,
            key=lambda x: (x.visible_count / label_mask.sum().item()) * (x.area / max_area),
            reverse=True,
        )[:topk]

        # We now focus exclusively on the best view
        best_stat = selected_stats[0]

        # Crop 1: Full-scene context crops (levels=0)
        vlm_full_params = CropParams(
            img_size=crop_params.img_size,
            levels=0,
            masked_crop=False,
            expansion_ratio=crop_params.expansion_ratio,
            dynamic_ratio=crop_params.dynamic_ratio,
            alpha_blend=0.0,
        )
        crop_full_dict = preprocess_crops(
            render_params, [best_stat], None, vlm_full_params, use_rendering
        )

        # Crop 2: Medium distance context crops (levels=1, expansion=3.0)
        vlm_medium_params = CropParams(
            img_size=crop_params.img_size,
            levels=1,
            masked_crop=False,
            expansion_ratio=3.0,
            dynamic_ratio=False,
            alpha_blend=0.0,
        )
        crop_medium_dict = preprocess_crops(
            render_params, [best_stat], None, vlm_medium_params, use_rendering
        )
        
        # Crop 3: Close-up zoom crops (levels=1, expansion=1.2)
        vlm_close_params = CropParams(
            img_size=crop_params.img_size,
            levels=1,
            masked_crop=False,
            expansion_ratio=1.2,
            dynamic_ratio=False,
            alpha_blend=0.0,
        )
        crop_close_dict = preprocess_crops(
            render_params, [best_stat], None, vlm_close_params, use_rendering
        )

        if "default" not in crop_full_dict or "default" not in crop_medium_dict or "default" not in crop_close_dict:
            continue

        raw_full, masks_full = crop_full_dict["default"]
        raw_medium, masks_medium = crop_medium_dict["default"]
        raw_close, masks_close = crop_close_dict["default"]
        
        # Apply visualization
        vis_full_img = (apply_vlm_visualization(raw_full[0].float() / 255.0, masks_full[0].squeeze() > 0, darken=True, red_outline=True) * 255).byte()
        vis_medium_img = (apply_vlm_visualization(raw_medium[0].float() / 255.0, masks_medium[0].squeeze() > 0, darken=True, red_outline=True) * 255).byte()
        vis_close_img = (apply_vlm_visualization(raw_close[0].float() / 255.0, masks_close[0].squeeze() > 0, darken=False, red_outline=True) * 255).byte()
        
        # Pack the 3 views into a single array to send together to Gemini
        vis_bundle = torch.stack([vis_full_img, vis_medium_img, vis_close_img])

        lang_model.to('cpu')
        torch.cuda.empty_cache()

        # Bypass local VLM and use Google API: Pass all 3 scale images in ONE single request
        images_list = []
        for i in range(vis_bundle.shape[0]):
            img = Image.fromarray(vis_bundle[i].permute(1, 2, 0).cpu().numpy())
            images_list.append(img)
            
        try:
            desc = call_gemini_robotics(prompt, images_list)
            # Since there is only 1 best perspective now (which generated the 3 crops),
            # we simply return a single description for this label, but keeping the loop
            # structure so the rest of the script (similarity check) functions perfectly.
            raw_descriptions = [desc]
            # Hard limit timer to stay securely below 30 RPM (Free Tier)
            time.sleep(2.1)
        except Exception as e:
            print(f"Gemini API error: {e}")
            raw_descriptions = ["Unidentified object."]
        
        # Strip "Description: " prefix
        descriptions = []
        for desc in raw_descriptions:
            if desc.startswith("Description:"):
                desc = desc[len("Description:"):].strip()
            descriptions.append(desc)
            
        lang_model.to('cuda')

        # Calculate text embeddings for the descriptions using the same lang_model
        with torch.no_grad():
            formatted_descriptions = [lang_model.prompt_template.format(desc) for desc in descriptions]
            text_embeds = lang_model.embed_text(formatted_descriptions, normalize=True)
            text_embeds = text_embeds.to(dtype=embeddings.dtype)
            
            # Average multiple text embeddings and re-normalize
            text_embedding = text_embeds.mean(dim=0)
            text_embedding = text_embedding / text_embedding.norm(p=2, dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().numpy()

        # Calculate distances and similarities (moved outside debug to allow global tracking)
        view_info = []
        with torch.no_grad():
            label_embedding_torch = embeddings[idx].to(text_embeds.device)
            
            # similarities via dot product (like in demo.py)
            raw_sim = text_embeds @ label_embedding_torch.unsqueeze(1)
            raw_sim = raw_sim.squeeze(1)
            
            cosine_distance = 1 - raw_sim
            
            # Rescale like in demo.py
            sim = lang_model.rescale(raw_sim)
            
            # Penalize "Visual Noise" so it gets the lowest possible similarity
            for i, desc in enumerate(descriptions):
                if desc.lower() == "visual noise":
                    sim[i] = -1000.0  # severely penalize visual noise

        for i in range(len(descriptions)):
            similarity_score = float(sim[i].cpu().item())
            cos_dist = float(cosine_distance[i].cpu().item())
            desc = descriptions[i]
            view_info.append({"description": desc, "similarity": similarity_score, "cosine_distance": cos_dist})
            
        # Sort view_info by similarity descending so the best match is first
        view_info.sort(key=lambda x: x["similarity"], reverse=True)
        
        if len(view_info) > 0:
            best_info = view_info[0]
            # Solo añadir si no es Visual Noise, o si es lo único que hay
            all_best_descriptions.append({
                "label_id": label_id,
                "description": best_info["description"],
                "similarity": best_info["similarity"],
                "cosine_distance": best_info["cosine_distance"]
            })

        if debug:
            label_debug_dir = debug_dir / str(label_id)
            label_debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the 3 scale images
            Image.fromarray(vis_bundle[0].permute(1, 2, 0).cpu().numpy()).save(label_debug_dir / "0_full_scene.png")
            Image.fromarray(vis_bundle[1].permute(1, 2, 0).cpu().numpy()).save(label_debug_dir / "1_medium_distance.png")
            Image.fromarray(vis_bundle[2].permute(1, 2, 0).cpu().numpy()).save(label_debug_dir / "2_extreme_close.png")
            
            # Save all info to a text file
            with open(label_debug_dir / "info.txt", "w") as f:
                for i, info in enumerate(view_info):
                    f.write(f"Rank {i}:\n")
                    f.write(f"  Similarity (CLIP): {info['similarity']:.4f}\n")
                    f.write(f"  Cosine Distance: {info['cosine_distance']:.4f}\n")
                    f.write(f"  Description: {info['description']}\n\n")

        results[label_id] = {
            "descriptions": descriptions,
            "embedding": embeddings[idx].numpy(),
            "text_embedding": text_embedding,
            "view_info": view_info if debug else None
        }
        torch.cuda.empty_cache()
        
        # Iterative Auto-Save so we don't lose progress if interrupted
        output_path = model_path / f"{lang_model.model_type}_descriptions.pth"
        torch.save(results, output_path)

    output_path = model_path / f"{lang_model.model_type}_descriptions.pth"
    torch.save(results, output_path)
    print(f"\nDescriptions saved to: {output_path}")
    
    # Save the global summary file
    summary_path = model_path / "vlm_best_descriptions_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== SUMMARY OF BEST DESCRIPTIONS ===\n\n")
        
        total_sim = 0.0
        total_cos_dist = 0.0
        
        for item in all_best_descriptions:
            f.write(f"Label ID: {item['label_id']}\n")
            f.write(f"Description: {item['description']}\n")
            f.write(f"Similarity (CLIP): {item['similarity']:.4f}\n")
            f.write(f"Cosine Distance: {item['cosine_distance']:.4f}\n")
            f.write("-" * 40 + "\n")
            
            total_sim += item["similarity"]
            total_cos_dist += item["cosine_distance"]
            
        n_items = len(all_best_descriptions) if len(all_best_descriptions) > 0 else 1
        avg_sim = total_sim / n_items
        avg_cos_dist = total_cos_dist / n_items
        
        f.write("\n=== GLOBAL METRICS ===\n")
        f.write(f"Total instances processed: {len(all_best_descriptions)}\n")
        f.write(f"Average Similarity (CLIP): {avg_sim:.4f}\n")
        f.write(f"Average Cosine Distance: {avg_cos_dist:.4f}\n")
        
    print(f"Global summary saved to: {summary_path}")

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
