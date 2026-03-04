import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips


def normalize_tensor(tensor, new_min=0.0, new_max=1.0):
    old_min, old_max = tensor.min(), tensor.max()
    if old_max == old_min:  
        return torch.full_like(tensor, new_min)
    return (tensor - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


class Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, delta=0.5, omega=0.2, reduction='mean', debug=False, device='cuda'):
        super(Loss, self).__init__()
        self.alpha = alpha      # MSE weight
        self.beta = beta        # MAE weight
        self.gamma = gamma      # Gradient weight
        self.delta = delta      # SSIM weight
        self.omega = omega      # LPIPS weight
        self.reduction = reduction
        self.debug = debug
        self.device = device
        self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device)

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
            print(f"Raw SSIM: {ssim_value:.6f}, SSIM Loss: {ssim_loss:.6f}")
        return ssim_loss

    def forward(self, output, target):
        if output.shape != target.shape:
            raise RuntimeError(f"Shape mismatch: output={output.shape}, target={target.shape}")

        output = output.to(self.device)
        target = target.to(self.device)

        mse = F.mse_loss(output, target, reduction=self.reduction)
        mae = F.l1_loss(output, target, reduction=self.reduction)
        gradient = self.gradient_loss(output, target)
        ssim_loss_val = self.ssim_loss(output, target)
        lpips_loss_val = self.lpips_loss(output, target).mean()

        total_loss = (self.alpha * mse +
                      self.beta * mae +
                      self.gamma * gradient +
                      self.delta * ssim_loss_val +
                      self.omega * lpips_loss_val)

        if self.debug:
            print(f"MSE: {mse.item():.6e}, MAE: {mae.item():.6e}, Grad: {gradient.item():.6e}, SSIM: {ssim_loss_val:.6e}, LPIPS: {lpips_loss_val.item():.6e}")
        return total_loss
