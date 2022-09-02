import torch
import torch.nn.functional as F
from torch import nn


class GroupNorm1D(nn.Module):
    def __init__(self, indim, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, indim)

    def forward(self, x):
        return self.gn(x.permute(1, 2, 0)).permute(2, 0, 1)


class GNActDWConv2d(nn.Module):
    def __init__(self, indim, gn_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(gn_groups, indim)
        self.conv = nn.Conv2d(indim,
                              indim,
                              5,
                              dilation=1,
                              padding=2,
                              groups=indim,
                              bias=False)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.gn(x)
        x = F.gelu(x)
        x = self.conv(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x


class ConvGN(nn.Module):
    def __init__(self, indim, outdim, kernel_size, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(indim,
                              outdim,
                              kernel_size,
                              padding=kernel_size // 2)
        self.gn = nn.GroupNorm(gn_groups, outdim)

    def forward(self, x):
        return self.gn(self.conv(x))

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
            self.gn3 = nn.GroupNorm(8, outdim)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
        
        self.gn1 = nn.GroupNorm(8, outdim)
        self.gn2 = nn.GroupNorm(8, outdim)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.gn1(r)

        r = self.conv2(F.relu(r))
        r = self.gn2(r)
 
        if self.downsample is not None:
            x = self.gn3(self.downsample(x))
         
        return x + r 
class ResGN(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.res1 = ResBlock(indim, outdim)
        self.res2 = ResBlock(outdim, outdim)


    def forward(self, x):
        return self.res2(self.res1(x))

        
def seq_to_2d(tensor, size_2d):
    h, w = size_2d
    _, n, c = tensor.size()
    tensor = tensor.view(h, w, n, c).permute(2, 3, 0, 1).contiguous()
    return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, batch_dim=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob):
        if drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = [1 for _ in range(x.ndim)]
        shape[self.batch_dim] = x.shape[self.batch_dim]
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class DropOutLogit(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropOutLogit, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_logit(x, self.drop_prob)

    def drop_logit(self, x, drop_prob):
        if drop_prob == 0. or not self.training:
            return x
        random_tensor = drop_prob + torch.rand(
            x.shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        mask = random_tensor * 1e+8 if (
            x.dtype == torch.float32) else random_tensor * 1e+4
        output = x - mask
        return output
