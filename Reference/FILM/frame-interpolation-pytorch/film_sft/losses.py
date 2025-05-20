# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips  # pip install lpips
import kornia.losses  # ✅ 使用 Kornia SSIM 替代 pytorch-msssim


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps ** 2)
        return loss.mean()


class SSIMLoss(nn.Module):
    def forward(self, pred, target):
        # Kornia SSIM loss: already returns a value in [0,1], lower is better
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        return kornia.losses.ssim_loss(pred, target, window_size=11, reduction='mean')


class LPIPSLoss(nn.Module):
    def __init__(self, net='vgg'):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net)
        for p in self.loss_fn.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return self.loss_fn(pred, target).mean()


def build_loss(name):
    """
    name: "l1+ssim", "charbonnier+lpips", etc.
    """
    name = name.lower()
    losses = []
    weights = []

    if "l1" in name:
        losses.append(L1Loss())
        weights.append(1.0)

    if "charbonnier" in name:
        losses.append(CharbonnierLoss())
        weights.append(1.0)

    if "ssim" in name:
        losses.append(SSIMLoss())  # ✅ 已使用 Kornia 实现
        weights.append(1.0)

    if "lpips" in name:
        losses.append(LPIPSLoss())
        weights.append(0.1)  # 一般设置较小权重

    return CombinedLoss(losses, weights)


class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, pred, target):
        total = 0
        for loss_fn, w in zip(self.losses, self.weights):
            total += w * loss_fn(pred, target)
        return total
