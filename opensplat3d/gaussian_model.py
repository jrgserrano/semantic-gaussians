from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from diff_gaussian_rasterization import MAX_FEATURE_DIM
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from opensplat3d.utils.general_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)
from opensplat3d.utils.scene_utils import BasicPointCloud, init_parameters
from opensplat3d.utils.sh_utils import RGB2SH


def build_covariance_from_scaling_rotation(
    rotation: torch.Tensor, scaling: torch.Tensor, scaling_modifier: float
):
    L = build_scaling_rotation(scaling * scaling_modifier, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class GaussianModel:
    def __init__(self) -> None:
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # color features
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)

        # other features
        self._features: torch.Tensor | None = torch.empty(0)

        # semantic opacity
        self._sem_opacity: torch.Tensor | None = torch.empty(0)

        # setup functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def num_points(self) -> int:
        return self._xyz.size(0)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_spherical_harmonics(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_features(self) -> torch.Tensor | None:
        return self._features

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.covariance_activation(
            self._rotation, self.get_scaling, scaling_modifier
        )

    def state_dict(self):
        return {
            "xyz": self._xyz,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "features": self._features,
        }

    def load_state_dict(self, state_dict: dict):
        self._xyz = state_dict["xyz"]
        self._scaling = state_dict["scaling"]
        self._rotation = state_dict["rotation"]
        self._opacity = state_dict["opacity"]
        self._features_dc = state_dict["features_dc"]
        self._features_rest = state_dict["features_rest"]
        self._features = state_dict["features"]


def create_from_pcd(
    pcd: BasicPointCloud,
    max_sh_degree: int,
    device: torch.device,
    with_features: bool = False,
    feature_dim: int = MAX_FEATURE_DIM,
    feature_init: str = "sh",
    static_xyz: bool = False,
):
    assert feature_dim <= MAX_FEATURE_DIM, (
        f"Requested feature dimension {feature_dim} exceeds maximum of {MAX_FEATURE_DIM}"
    )
    model = GaussianModel()
    fused_point_cloud = pcd.points.float().to(device)
    fused_color = RGB2SH(pcd.colors.float().to(device))
    features = torch.zeros(
        (fused_color.size(0), 3, (max_sh_degree + 1) ** 2),
        dtype=torch.float,
        device=device,
    )
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation:", fused_point_cloud.size(0))

    assert fused_point_cloud.is_cuda, "point cloud points tensor is not on GPU"
    # The point clouds from the Astra robot can be quite sparse.
    # We clamp the distance to prevent the initial Gaussians from being massive and covering the entire screen.
    dist2 = torch.clamp(
        distCUDA2(fused_point_cloud).float().to(device), min=0.0000001, max=0.0001
    )
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
    rots[:, 0] = 1

    # Revert to standard 3DGS 0.1 opacity. Huge opacities cause ray-blocking if scales are large.
    opacities = inverse_sigmoid(
        0.1 * torch.ones(
            (fused_point_cloud.shape[0], 1), dtype=torch.float, device=device
        )
    )

    model._xyz = nn.Parameter(fused_point_cloud, requires_grad=not static_xyz)
    model._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
    model._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
    if with_features:
        # If the point cloud carries SAM instance labels, seed features with
        # per-instance embeddings so that Gaussians of the same object start
        # close together in feature space (warm-start for clustering).
        if pcd.instance_ids is not None:
            ids = pcd.instance_ids.to(device)   # (N,)
            unique_ids = torch.unique(ids)
            # Build a deterministic embedding per unique id (seeded by id value)
            # so that different runs are consistent and different instances are
            # spread apart in feature space.
            rng = torch.Generator(device=device)
            id_to_emb = {}
            for uid in unique_ids.tolist():
                rng.manual_seed(int(uid) & 0xFFFFFFFF)
                id_to_emb[int(uid)] = torch.randn(feature_dim, generator=rng, device=device)
            feat_init = torch.stack([id_to_emb[int(i.item())] for i in ids])  # (N, D)
            # Unlabelled points (id=-1) stay random
            unlabelled = (ids == -1)
            if unlabelled.any():
                rng2 = torch.Generator(device=device)
                feat_init[unlabelled] = torch.randn(
                    (unlabelled.sum(), feature_dim), generator=rng2, device=device
                )
            model._features = nn.Parameter(feat_init * 0.1)
            print(f"  [Instance Init] Warm-started {(~unlabelled).sum()}/{len(ids)} "
                  f"Gaussian features from {len(unique_ids)-1} SAM instances.")
        else:
            model._features = nn.Parameter(
                init_parameters(
                    (fused_point_cloud.size(0), feature_dim), feature_init, device
                )
            )

    else:
        model._features = None

    model._scaling = nn.Parameter(scales)
    model._rotation = nn.Parameter(rots)
    model._opacity = nn.Parameter(opacities)
    return model


def create_from_ply(
    ply_path: Path,
    max_sh_degree: int,
    device: torch.device,
    only_feature_grads: bool = False,
    feature_dim: int = MAX_FEATURE_DIM,
):
    # only called for loading inference model
    plydata = PlyData.read(str(ply_path))

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
    )

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    feat_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("feat_")
    ]
    features: npt.NDArray | None = None
    if len(feat_names) > 0:
        feat_names = sorted(feat_names, key=lambda x: int(x.split("_")[-1]))
        features = np.zeros((xyz.shape[0], len(feat_names)))
        for idx, attr_name in enumerate(feat_names):
            features[:, idx] = np.asarray(plydata.elements[0][attr_name])

    model = GaussianModel()
    model._xyz = nn.Parameter(
        torch.tensor(xyz, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )
    model._features_dc = nn.Parameter(
        torch.tensor(features_dc, dtype=torch.float32, device=device)
        .transpose(1, 2)
        .contiguous(),
        requires_grad=not only_feature_grads,
    )
    model._features_rest = nn.Parameter(
        torch.tensor(features_extra, dtype=torch.float32, device=device)
        .transpose(1, 2)
        .contiguous(),
        requires_grad=not only_feature_grads,
    )
    model._opacity = nn.Parameter(
        torch.tensor(opacities, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )
    model._scaling = nn.Parameter(
        torch.tensor(scales, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )
    model._rotation = nn.Parameter(
        torch.tensor(rots, dtype=torch.float32, device=device),
        requires_grad=not only_feature_grads,
    )

    allow_mismatch = False
    if features is not None and (allow_mismatch or feature_dim == features.shape[1]):
        model._features = nn.Parameter(
            torch.tensor(features, dtype=torch.float32, device=device)
        )
    elif not allow_mismatch:
        if features is not None:
            print(
                f"Feature dimension mismatch, expected {feature_dim} but got {features.shape[1]}"
            )
        model._features = nn.Parameter(
            RGB2SH(
                torch.rand(
                    (xyz.shape[0], feature_dim),
                    dtype=torch.float32,
                    device=device,
                )
            )
        )

    return model


def construct_list_of_attributes(model: GaussianModel):
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(model._features_dc.size(1) * model._features_dc.size(2)):
        attributes.append(f"f_dc_{i}")
    for i in range(model._features_rest.size(1) * model._features_rest.size(2)):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(model._scaling.size(1)):
        attributes.append(f"scale_{i}")
    for i in range(model._rotation.size(1)):
        attributes.append(f"rot_{i}")
    if model._features is not None:
        for i in range(model._features.size(1)):
            attributes.append(f"feat_{i}")
    return attributes


def save_ply(model: GaussianModel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz: npt.NDArray = model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = (
        model._features_dc.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    f_rest = (
        model._features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = model._opacity.detach().cpu().numpy()
    scale = model._scaling.detach().cpu().numpy()
    rotation = model._rotation.detach().cpu().numpy()
    features = None
    if model._features is not None:
        features = model._features.detach().cpu().numpy()

    dtype_full = [(a, "f4") for a in construct_list_of_attributes(model)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = (xyz, normals, f_dc, f_rest, opacities, scale, rotation)
    if features is not None:
        attributes = attributes + (features,)
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(str(path))


def save_gaussians(gaussians: GaussianModel, model_path: Path, iteration: int):
    pcd_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    save_ply(gaussians, pcd_path)


def sub_gaussians(gaussians: GaussianModel, mask: torch.BoolTensor | torch.LongTensor):
    state_dict = gaussians.state_dict()
    for key in state_dict.keys():
        if state_dict[key] is not None:
            state_dict[key] = state_dict[key][mask]
    new_gaussians = GaussianModel()
    new_gaussians.load_state_dict(state_dict)
    return new_gaussians
