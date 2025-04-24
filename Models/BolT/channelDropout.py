import torch
from torch import nn
class ChannelDropout(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        B, T, C = x.shape
        mask = torch.rand(B, C, device=x.device) > self.drop_prob  # (B, C)
        mask = mask.float().unsqueeze(1)  # (B, 1, C)
        return x * mask  # broadcast across time
