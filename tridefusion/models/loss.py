import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

def normalize_tensor(tensor, new_min=0.0, new_max=1.0):
    old_min, old_max = tensor.min(), tensor.max()
    return (tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.05, reduction='mean', debug=False):
        """
        Custom loss function combining MSE, gradient loss, and SSIM loss.
        
        Args:
            alpha (float): Weight for MSE loss.
            beta (float): Weight for gradient loss.
            gamma (float): Weight for SSIM loss.
            reduction (str): Reduction method for MSE and gradient loss ('mean' or 'sum').
            debug (bool): If True, print intermediate loss values during training.
        """
        super(Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.debug = debug

    def gradient_loss(self, pred, target):
        pred_grad_x = torch.diff(pred, dim=3)
        target_grad_x = torch.diff(target, dim=3)
        pred_grad_y = torch.diff(pred, dim=2)
        target_grad_y = torch.diff(target, dim=2)
        pred_grad_x = F.pad(pred_grad_x, (0, 1))
        target_grad_x = F.pad(target_grad_x, (0, 1))
        pred_grad_y = F.pad(pred_grad_y, (0, 0, 0, 1))
        target_grad_y = F.pad(target_grad_y, (0, 0, 0, 1))
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x, reduction=self.reduction)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y, reduction=self.reduction)
        return grad_loss_x + grad_loss_y

    def ssim_loss(self, pred, target):
        pred = normalize_tensor(pred, 0, 1)
        target = normalize_tensor(target, 0, 1)
        ssim_value = ssim(pred, target, data_range=1.0)
        ssim_loss = 1 - ssim_value  

        if self.debug:
            print(f"Raw SSIM Value: {ssim_value:.6f}, SSIM Loss: {ssim_loss:.6f}")
        return ssim_loss


    def forward(self, output, target):
        mse = F.mse_loss(output, target, reduction=self.reduction)
        gradient = self.gradient_loss(output, target)
        ssim = self.ssim_loss(output, target)
        total_loss = self.alpha * mse + self.beta * gradient + self.gamma * ssim
        if self.debug:
            print(f"MSE Loss: {mse.item():.6f}, Gradient Loss: {gradient.item():.6f}, SSIM Loss: {ssim:.6f}")
        return total_loss
