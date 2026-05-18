import torch
import torch.nn as nn

class PoseOptimizer(nn.Module):
    def __init__(self, num_cameras, device="cuda"):
        super().__init__()
        # 3 para rotación (ejes-ángulo), 3 para traslación
        self.offsets = nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        
    def forward(self, cam_idx, original_R, original_T):
        """
        Aplica el offset actual a la rotación y traslación originales.
        """
        delta = self.offsets[cam_idx]
        delta_R = self.axis_angle_to_matrix(delta[:3])
        delta_T = delta[3:]
        
        # Nueva R = delta_R @ original_R
        # Nueva T = delta_R @ original_T + delta_T
        new_R = delta_R @ original_R
        new_T = (delta_R @ original_T.unsqueeze(-1)).squeeze(-1) + delta_T
        
        return new_R, new_T

    def axis_angle_to_matrix(self, v):
        # Implementación de Rodrigues simplificada
        theta = torch.norm(v)
        if theta < 1e-6:
            return torch.eye(3, device=v.device)
        k = v / theta
        K = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], device=v.device)
        return torch.eye(3, device=v.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

    def get_smoothness_loss(self):
        """
        Calcula la pérdida de suavidad temporal (Motion Prior).
        Penaliza diferencias grandes entre los offsets de cámaras contiguas.
        """
        if self.offsets.shape[0] < 2:
            return torch.tensor(0.0, device=self.offsets.device)
            
        # Diferencia entre frames consecutivos
        diff = self.offsets[1:] - self.offsets[:-1]
        return torch.mean(diff**2)
