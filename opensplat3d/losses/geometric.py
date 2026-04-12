import torch


def get_erank_loss(scaling: torch.Tensor, eps: float = 1e-8):
    """
    Computes eRank loss for Gaussian Splatting as proposed in FeatureSLAM.
    Forces Gaussians to be more isotropic (less needle-like).
    """
    # Inverse of scaling is handled inside but we assume scaling is in 'eagar' space (not log)
    # The actual geometric property is Shannon entropy of the scales.
    s2 = scaling.pow(2)
    p = s2 / (s2.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    erank = torch.exp(entropy)
    # Penalize low eRank (needle shapes). Low eRank = high penalty.
    # We want eRank to be closer to 3 (isotropic in 3D).
    return -torch.log(erank - 1 + eps).mean()


def get_thinness_loss(scaling: torch.Tensor, eps: float = 1e-8):
    """
    Computes Thinness loss for Gaussian Splatting as proposed in FeatureSLAM.
    Penalizes excessively thin 2D planar Gaussian shapes.
    """
    s_min = scaling.min(dim=-1)[0]
    # Minimize the inverse of the smallest scale to avoid vanishing thickness.
    return (1.0 / (s_min + eps)).mean()
