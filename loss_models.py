import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import get_device

class VanillaMultiLoss(nn.Module):
    def __init__(self, n_losses: int): 
        super().__init__()
        self.loss_term_scaling = nn.Parameter(torch.ones(n_losses, device=get_device()))

    def forward(self, losses: list) -> torch.Tensor:
        total_loss = torch.zeros(1, device=losses[0].device, dtype=losses[0].dtype)
        for i, loss in enumerate(losses):
            scaling = F.softplus(self.loss_term_scaling[i])  # ensure > 0

            total_loss += scaling * loss
 
        return total_loss

class MultiNoiseLoss(nn.Module):
    """
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (Kendall et al; CVPR 2018).
    """
    def __init__(self, n_losses: int):
        super(MultiNoiseLoss, self).__init__()
        self.raw_noise_params = nn.Parameter(torch.randn(n_losses, device=get_device())) # learnable normal noise
    
    def forward(self, losses: list) -> torch.Tensor:
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)

        Each loss coeff is of the form: :math:`\frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i)`
        Total loss: :math:`\ell = \sum_{i=1}^{k} \left\[ \frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i) \right\]`
        """
        total_loss = torch.zeros(1, device=losses[0].device, dtype=losses[0].dtype)
        for i, loss in enumerate(losses):
            noise_i = F.softplus(self.raw_noise_params[i]) + 1e-6  # ensure > 0
            total_loss += (1 / noise_i ** 2) * loss + torch.log(noise_i)

        return total_loss
