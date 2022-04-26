from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDN(nn.Module):
    def __init__(self, p=2):
        super(KDN, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]


class CE_MSE(nn.Module):
    def __init__(self, alpha=0.5, T=4.0):
        super(CE_MSE, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
        self.T = nn.Parameter(torch.Tensor([T]))
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, logits_s, logits_t):
        p_s = logits_s / self.T
        p_t = logits_t / self.T
        mse_temp_loss = F.mse_loss(p_s,p_t)
        ce_loss = self.ce_criterion(logits_s, logits_t)
        return (1-self.alpha) * ce_loss + (self.alpha) * mse_temp_loss
