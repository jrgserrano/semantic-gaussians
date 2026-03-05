import groundingdino.datasets.transforms as T
import numpy.typing as npt
import torch
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from segment_anything import SamPredictor


def load_model_hf(
    repo_id: str,
    filename: str,
    ckpt_config_filename: str,
):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu", weights_only=False)
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


# Copied from https://github.com/lkeab/gaussian-grouping/blob/main/ext/grounded_sam.py#L58
@torch.no_grad()
def grounded_sam_output(
    grounding_dino: GroundingDINO,
    sam_predictor: SamPredictor,
    text_prompt: str,
    image: npt.NDArray,
    box_threshold: float = 0.3,
    text_threshold: float = 0.45,
    device: torch.device | str = "cuda",
):
    image_source = image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img, _ = transform(Image.fromarray(image_source), None)

    boxes, _, _ = predict(
        model=grounding_dino,
        image=img,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # set image
    sam_predictor.set_image(image_source)
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    if len(boxes_xyxy) > 0:
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        ).to(device)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    else:
        masks = torch.zeros((1, 1, H, W)).cuda()

    return torch.sum(masks, dim=0).squeeze().bool()
