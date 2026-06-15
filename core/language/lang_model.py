import os
import typing

import numpy.typing as npt
import open_clip
import torch
from PIL import Image
from torchvision.transforms import Compose
from transformers.models.siglip import SiglipModel, SiglipProcessor

from core.language.masqclip import MasQCLIP

EMBED_DIM = {
    "clip": 512,
    "siglip": 1152,
    "masqclip": 768,
}

IMG_SIZE = {
    "clip": (224, 224),
    "siglip": (384, 384),
    "masqclip": (336, 336),
}

WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", ".")


class LanguageModel:
    def __init__(self, model: str):
        self.model_type = model
        self.clip_model: open_clip.CLIP | None = None
        self.clip_tokenizer: open_clip.SimpleTokenizer | None = None
        self.clip_preprocess: Compose | None = None
        self.siglip_model: SiglipModel | None = None
        self.siglip_processor: SiglipProcessor | None = None
        self.masqclip_model: MasQCLIP | None = None

        self.embed_dim: int = 0
        self.dtype = torch.float32

        if self.model_type == "clip":
            self.dtype = torch.float16
            model_type = "ViT-B-16"
            pretrained = "laion2b_s34b_b88k"
            precision = "fp16"
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                model_type,
                pretrained=pretrained,
                precision=precision,
            )
            tokenizer = open_clip.get_tokenizer(model_type)
            assert isinstance(clip_model, open_clip.CLIP)
            assert isinstance(tokenizer, open_clip.SimpleTokenizer)
            assert isinstance(clip_preprocess, Compose)
            clip_model.cuda()
            self.clip_model = clip_model.eval()
            self.clip_tokenizer = tokenizer
            self.clip_preprocess = clip_preprocess
        elif self.model_type == "siglip":
            self.dtype = torch.float16
            siglip_model = SiglipModel.from_pretrained(
                "google/siglip-so400m-patch14-384",
                attn_implementation="sdpa",
                torch_dtype=torch.float16,
                device_map="cuda",
            )
            siglip_processor = SiglipProcessor.from_pretrained(
                "google/siglip-so400m-patch14-384"
            )
            assert isinstance(siglip_model, SiglipModel)
            assert isinstance(siglip_processor, SiglipProcessor)
            self.siglip_model = siglip_model
            self.siglip_processor = siglip_processor
        elif self.model_type == "masqclip":
            self.dtype = torch.float32
            masqclip_model = MasQCLIP(["ViT-L/14@336px"])
            masqclip_model.from_pretrained(
                f"{WORKSPACE_PATH}/ckpts/MasQCLIP/base_novel.pth"
            )
            assert isinstance(masqclip_model, MasQCLIP)
            masqclip_model.cuda()
            masqclip_model.eval()
            self.masqclip_model = masqclip_model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.embed_dim = EMBED_DIM[self.model_type]
        self.img_size = IMG_SIZE[self.model_type]

        self.prompt_template = "an image of {}"

    def to(self, device):
        if self.model_type == "clip":
            if getattr(self, "clip_model", None) is not None:
                self.clip_model.to(device)
        elif self.model_type == "siglip":
            if getattr(self, "siglip_model", None) is not None:
                self.siglip_model.to(device)
        elif self.model_type == "masqclip":
            if getattr(self, "masqclip_model", None) is not None:
                self.masqclip_model.to(device)

    @torch.no_grad()
    def preprocess_images(self, images: torch.Tensor | npt.NDArray) -> torch.Tensor:
        # images: (B, C, H, W) if torch, (B, H, W, C) if numpy
        assert images.ndim == 4, f"Expected 4D tensor, got {images.ndim}D"
        if isinstance(images, torch.Tensor):
            images = images.permute(0, 2, 3, 1).contiguous()
            images = images.cpu().numpy()

        assert images.shape[-1] == 3, f"Expected 3 channels, got {images.shape[3]}"
        if self.model_type == "clip":
            assert self.clip_preprocess is not None
            return torch.stack(
                [
                    typing.cast(torch.Tensor, self.clip_preprocess(Image.fromarray(x)))
                    for x in images
                ]
            )

        elif self.model_type == "siglip":
            assert self.siglip_processor is not None
            return self.siglip_processor(images=[Image.fromarray(x) for x in images])[
                "pixel_values"
            ]
        elif self.model_type == "masqclip":
            assert self.masqclip_model is not None
            return self.masqclip_model.preprocess_images(images)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @torch.no_grad()
    def embed_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        # images need to be preprocessed before calling this function
        # images = self.preprocess_images(images)
        if self.model_type == "clip":
            assert self.clip_model is not None
            image_embed = self.clip_model.encode_image(images, normalize=normalize)
        elif self.model_type == "siglip":
            assert self.siglip_model is not None
            image_embed = self.siglip_model.get_image_features(
                typing.cast(torch.FloatTensor, images)
            )
            if normalize:
                image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True)
        elif self.model_type == "masqclip":
            assert self.masqclip_model is not None
            assert masks is not None, "Masks must be provided for MasQCLIP"
            image_embed = self.masqclip_model.get_image_embedding(
                images, masks, normalize=normalize
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return image_embed

    @torch.no_grad()
    def embed_text(self, prompts: list[str], normalize: bool = False) -> torch.Tensor:
        if self.model_type == "clip":
            assert self.clip_model is not None and self.clip_tokenizer is not None
            tok_phrases = torch.cat(
                [self.clip_tokenizer(phrase) for phrase in prompts]
            ).to(self.clip_model.logit_scale.device)
            text_embed = self.clip_model.encode_text(tok_phrases, normalize=normalize)
        elif self.model_type == "siglip":
            assert self.siglip_model is not None and self.siglip_processor is not None
            tok_phrases = self.siglip_processor(
                text=prompts, padding="max_length", return_tensors="pt"
            )
            tok_phrases = tok_phrases.to(self.siglip_model.device)
            text_embed = self.siglip_model.get_text_features(**tok_phrases)
            if normalize:
                text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
        elif self.model_type == "masqclip":
            assert self.masqclip_model is not None
            text_embed = self.masqclip_model.get_text_embedding(
                prompts, normalize=normalize
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return text_embed

    @torch.no_grad()
    def rescale(self, similarity: torch.Tensor) -> torch.Tensor:
        if self.model_type == "clip":
            assert self.clip_model is not None
            rescaled = self.clip_model.logit_scale.exp() * similarity
            if self.clip_model.logit_bias is not None:
                rescaled += self.clip_model.logit_bias
        elif self.model_type == "siglip":
            assert self.siglip_model is not None
            rescaled = (
                self.siglip_model.logit_scale.exp() * similarity
                + self.siglip_model.logit_bias
            )
        elif self.model_type == "masqclip":
            assert self.masqclip_model is not None
            rescaled = self.masqclip_model.clip.logit_scale.exp() * similarity
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return rescaled

    @torch.no_grad()
    def activation(self, similarity: torch.Tensor) -> torch.Tensor:
        if self.model_type == "clip":
            return torch.softmax(similarity, dim=-1)
        elif self.model_type == "siglip":
            return torch.sigmoid(similarity)
        elif self.model_type == "masqclip":
            return torch.softmax(similarity, dim=-1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
