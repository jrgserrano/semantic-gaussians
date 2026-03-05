import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib import animation, colormaps
from matplotlib.colors import ListedColormap, Normalize
from PIL import Image, ImageEnhance
from sklearn.decomposition import PCA


def images2video(
    image_array: npt.NDArray | torch.Tensor,
    interval: int = 99,
    dpi: float = 72.0,
    norm: Normalize | None = None,
):
    ypixels, xpixels = image_array.shape[1:3]
    fig = plt.figure(figsize=(xpixels / dpi, ypixels / dpi), dpi=dpi)
    im = plt.figimage(image_array[0], norm=None)

    def animate(i):
        im.set_array(image_array[i])
        if norm is None:
            im.autoscale()
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(image_array),
        interval=interval,
        repeat_delay=1,
        repeat=True,
    )
    return anim


def pca(features: npt.NDArray, n_components: int = 3, normalize: bool = True):
    assert features.shape[-1] >= n_components, (
        "Feature dimension must be greater or equal to number of components."
    )
    if features.shape[-1] > n_components:
        pca = PCA(n_components=n_components, random_state=42)
        pca_result = pca.fit_transform(features)
    else:
        pca_result = features
    if normalize:
        pca_result: npt.NDArray = (pca_result - pca_result.min()) / (
            pca_result.max() - pca_result.min()
        )
    return pca_result


def feature_image_pca_3d(features: npt.NDArray):
    # Input features shape: (16, H, W)
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.reshape(features.shape[0], -1).T
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped)
    pca_result = pca_result.reshape(H, W, 3)
    pca_normalized: npt.NDArray = (
        255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    )
    return pca_normalized.astype(np.uint8)


def get_seg_cmap(n: int = 256, black_ignore: bool = True):
    cm_prism = colormaps.get_cmap("prism")
    newcolors = cm_prism(np.linspace(0, 1, n))
    if black_ignore:
        newcolors[0, :] = np.array([0, 0, 0, 1], dtype=np.float32)
    newcmap = ListedColormap(newcolors)
    return newcmap


def get_cluster_color(labels: npt.NDArray, seed: int = 42) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    random_colors = rng.random((labels.shape[0], 3)).astype(dtype=np.float32)
    cluster_color = torch.from_numpy(random_colors[labels])
    cluster_color[labels == -1] = 0
    return cluster_color


def enhance_image(image: npt.NDArray, contrast: float = 1.5, saturation: float = 2.0):
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    im_pil = Image.fromarray(image)
    im_pil = ImageEnhance.Contrast(im_pil).enhance(contrast)
    im_pil = ImageEnhance.Color(im_pil).enhance(saturation)
    return np.array(im_pil)
