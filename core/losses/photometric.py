import torch
import torch.nn.functional as F

from core.utils.loss_utils import create_window


# @torch.compile()
def l1_loss(network_output: torch.Tensor, gt: torch.Tensor):
    return torch.abs((network_output - gt)).mean()


@torch.compile()
def l2_loss(network_output: torch.Tensor, gt: torch.Tensor):
    return ((network_output - gt) ** 2).mean()


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_pytorch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


try:
    from fused_ssim import fused_ssim

    def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        size_average: bool = True,
    ) -> torch.Tensor:
        if img1.ndim < 4:
            img1 = img1[None]
        if img2.ndim < 4:
            img2 = img2[None]
        return fused_ssim(img1, img2)

    print("Using fused SSIM")
except ImportError:
    print("Using slow pytorch SSIM")
    ssim = ssim_pytorch
