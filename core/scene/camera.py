import torch
from torch import nn
from torchvision.transforms.v2 import InterpolationMode, Resize
from tqdm.contrib.concurrent import thread_map

from core.utils.camera_utils import get_projection_matrix, get_world2view
from core.utils.mask_utils import mask_consecutive_labels
from core.utils.scene_utils import CameraInfo


class Camera(nn.Module):
    def __init__(
        self,
        uid: int,
        name: str,
        R: torch.FloatTensor,
        T: torch.FloatTensor,
        fovX: float,
        fovY: float,
        image: torch.FloatTensor,  # CHW
        alpha_mask: torch.FloatTensor | None,
        masks: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        cx: float | None = None,
        cy: float | None = None,
        normal: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.uid = uid
        self.name = name
        self.register_buffer("R", R)
        self.register_buffer("T", T)
        
        self.register_buffer("original_R", R.clone())
        self.register_buffer("original_T", T.clone())
        
        self.fovX = fovX
        self.fovY = fovY
        self.znear = 0.01
        self.zfar = 100.0

        self.original_image = image.clamp(0.0, 1.0)  # CHW
        self.image_width = self.original_image.size(2)
        self.image_height = self.original_image.size(1)

        self.masks = masks
        self.original_depth = depth
        self.cx = cx
        self.cy = cy
        self.normal = normal

        if alpha_mask is not None:
            self.original_image *= alpha_mask

        world_view_transform = get_world2view(R, T).transpose(0, 1)
        self.register_buffer("world_view_transform", world_view_transform)
        self.world_view_transform: torch.Tensor = self.world_view_transform

        projection_matrix = get_projection_matrix(
            znear=self.znear, zfar=self.zfar, fovX=fovX, fovY=fovY,
            cx=cx, cy=cy, W=self.image_width, H=self.image_height
        ).transpose(0, 1)
        self.register_buffer("projection_matrix", projection_matrix)
        self.projection_matrix: torch.Tensor = self.projection_matrix

        self.register_buffer(
            "full_proj_transform",
            (
                self.world_view_transform.unsqueeze(0).bmm(
                    self.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0),
        )
        self.full_proj_transform: torch.Tensor = self.full_proj_transform

        self.register_buffer(
            "camera_center", self.world_view_transform.inverse()[3, :3]
        )
        self.camera_center: torch.Tensor = self.camera_center

    def update_matrices(self):
        # Recalculamos la matriz World-to-View (Vista)
        world_view_transform = get_world2view(self.R, self.T).transpose(0, 1)
        self.world_view_transform = world_view_transform.to(self.R.device)
        # Recalculamos la matriz de Proyección (por si acaso han cambiado FOV o centro)
        projection_matrix = get_projection_matrix(
            znear=self.znear, zfar=self.zfar, fovX=self.fovX, fovY=self.fovY,
            cx=self.cx, cy=self.cy, W=self.image_width, H=self.image_height
        ).transpose(0, 1)
        self.projection_matrix = projection_matrix.to(self.R.device)
        # La matriz final es la combinación de ambas
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        # El centro de la cámara es la inversa de la vista
        self.camera_center = self.world_view_transform.inverse()[3, :3]


WARNED = False


def to_camera(
    cam_info: CameraInfo,
    resolution: int,
    resolution_scale: float,
):
    orig_h, orig_w = cam_info.image.shape[:2]

    if resolution in [1, 2, 4, 8]:
        new_resolution = (
            round(orig_h / (resolution_scale * resolution)),
            round(orig_w / (resolution_scale * resolution)),
        )
    else:  # should be a type that converts to float
        if resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / resolution

        scale = float(global_down) * float(resolution_scale)
        new_resolution = (int(orig_h / scale), int(orig_w / scale))

    resizer = Resize(new_resolution)
    resized_image_rgb: torch.FloatTensor = resizer(
        (cam_info.image.float() / 255.0).permute(2, 0, 1)
    )
    gt_image: torch.FloatTensor = resized_image_rgb[:3, ...]  # type: ignore

    alpha_mask: torch.FloatTensor | None = None
    if resized_image_rgb.size(0) == 4:
        alpha_mask = resized_image_rgb[3:4, ...]  # type: ignore

    gt_masks: torch.Tensor | None = None
    if cam_info.masks is not None:
        mask_resizer = Resize(new_resolution, interpolation=InterpolationMode.NEAREST)
        old_device = cam_info.masks.device
        masks = cam_info.masks.cuda()
        labels: torch.Tensor = torch.unique(masks)
        if masks.ndim == 2:  # merged masked
            masks = mask_resizer(masks.unsqueeze(0)).squeeze(0)
        else:
            masks = mask_resizer(masks)

        # NOTE some masks can be very small and after resizing they are lost, therefore the labels are not consecutive anymore.
        labels2: torch.Tensor = torch.unique(masks)
        if labels.shape != labels2.shape:
            masks = mask_consecutive_labels(masks)
        gt_masks = masks.to(old_device)

    gt_depth: torch.Tensor | None = None
    if cam_info.depth is not None:
        depth_resizer = Resize(new_resolution, interpolation=InterpolationMode.BILINEAR)
        # Depth is usually HxW. We need 1xHxW to resize.
        depth_tensor = cam_info.depth.unsqueeze(0)
        gt_depth = depth_resizer(depth_tensor).squeeze(0)

    gt_normal: torch.Tensor | None = None
    if cam_info.normal is not None:
        normal_resizer = Resize(new_resolution, interpolation=InterpolationMode.BILINEAR)
        gt_normal = normal_resizer(cam_info.normal)

    return Camera(
        uid=cam_info.uid,
        name=cam_info.image_name,
        R=cam_info.R,
        T=cam_info.T,
        fovX=cam_info.fovX,
        fovY=cam_info.fovY,
        image=gt_image,
        alpha_mask=alpha_mask,
        masks=gt_masks,
        depth=gt_depth,
        cx=cam_info.cx * (new_resolution[1] / orig_w) if cam_info.cx is not None else None,
        cy=cam_info.cy * (new_resolution[0] / orig_h) if cam_info.cy is not None else None,
        normal=gt_normal,
    )


def to_cameras(
    cam_infos: list[CameraInfo],
    resolution_scale: float,
    resolution: int,
    device: torch.device,
    progbar: bool = False,
) -> list[Camera]:
    data = enumerate(cam_infos)

    def _wrapper(args: tuple[int, CameraInfo]) -> Camera:
        _, c = args
        return to_camera(c, resolution, resolution_scale).to(device)

    return thread_map(_wrapper, data, total=len(cam_infos), disable=not progbar)


class ViewerCamera:
    def __init__(
        self,
        width: int,
        height: int,
        fovX: float,
        fovY: float,
        znear: float,
        zfar: float,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
    ):
        self.image_width = width
        self.image_height = height
        self.fovX = fovX
        self.fovY = fovY
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
