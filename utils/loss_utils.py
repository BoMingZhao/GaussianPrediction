#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def motion_loss(gt_xyz, pred_xyz):
    loss = 0.
    for i in range(gt_xyz.shape[1]):
        gt = gt_xyz[:, i, ...] # (16, 100, 3)
        pred = pred_xyz[:, i, ...]
        # loss += l2_loss(pred, gt)
        loss += l1_loss(pred, gt) / (i + 5)
        # loss = l1_loss(pred, gt)
    return loss / gt_xyz.shape[1]

def iso_loss(ut, canonical_ut, k=20, lamda=2000):
    if len(ut.shape) == 2:
        N = ut.shape[0]
        ut = ut.unsqueeze(0)
    else:
        N = ut.shape[1]
    B = ut.shape[0]
    # knn = KNN(k = k + 1, transpose_mode=True)

    dists, uj_idx, _ = knn_points(ut.unsqueeze(0), ut.unsqueeze(0), K=k+1)
    # _, uj_idx, _, _ = frnn.frnn_grid_points(ut, ut, K=k+1, r=0.08)
    uj = []
    for b in range(B):
        uj += [ut[b, uj_idx[b, ..., 1:].contiguous().view([-1])]] # [N*k, 3]
    uj = torch.stack(uj)
    first_ut = canonical_ut
    _, first_uj_idx, _ = knn_points(first_ut.unsqueeze(0), first_ut.unsqueeze(0), K=k+1)
    # _, first_uj_idx, _, _  = frnn.frnn_grid_points(first_ut.unsqueeze(0), first_ut.unsqueeze(0), K=k+1, r=0.08)
    first_uj = first_ut[first_uj_idx[..., 1:].contiguous().view([-1])]

    # weights = torch.exp(-lamda * torch.norm(first_uj - first_ut.repeat(k, 1))**2)
    '''l_iso = (torch.abs(torch.norm(first_uj - first_ut.repeat(k, 1)) 
                      - torch.norm(uj - ut.repeat(k, 1))) * weights) / (k * N)'''
    l_iso = (torch.abs(torch.norm(first_uj - first_ut.repeat(k, 1)) - torch.norm(uj - ut.repeat(1, k, 1)))) / (k * N)

    return l_iso

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)