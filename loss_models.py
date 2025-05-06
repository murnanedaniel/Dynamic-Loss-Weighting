import torch
import torch.nn as nn

from utils.utils import get_device

class VanillaMultiLoss(nn.Module):
    def __init__(self, n_losses: int): 
        super().__init__()
        self.loss_term_scaling = nn.Parameter(torch.ones(n_losses, device=get_device()))

    def forward(self, losses: list) -> torch.tensor:
        total_loss = 0
        for i, loss in enumerate(losses):
            scaling = self.loss_term_scaling[i]
            
            loss = loss.squeeze() # ensures loss is scalar (shape []); handling loss tensor shapes like [1,1]
            
            total_loss += scaling * loss
 
        return total_loss
 
class MultiNoiseLoss(nn.Module):
    """
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (Kendall et al; CVPR 2018).
    """
    def __init__(self, n_losses: int):
        super(MultiNoiseLoss, self).__init__()
        self.noise_params = nn.Parameter(torch.rand(n_losses, device=get_device()))
    
    def forward(self, losses: list) -> torch.tensor:
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)

        Each loss coeff is of the form: :math:`\frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i)`
        Total loss: :math:`\ell = \sum_{i=1}^{k} \left\[ \frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i) \right\]`
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            
            loss = loss.squeeze() # ensures loss is scalar (shape []); handling loss tensor shapes like [1,1]
            total_loss += (1/torch.square(self.noise_params[i]))*loss + torch.log(self.noise_params[i])
        
        return total_loss
