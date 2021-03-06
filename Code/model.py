# This part is establishing AlphaNet-v1 model with self defined layers
# Input: torch.Size([1000, 1, 9, 30])
# Output: torch.Size([1000])
import torch
from torch import nn
from self_defined_layers import *


# Define a custom layer from a given function
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# Model
class AlphaNet_v1(nn.Module):
    def truncated_normal_(self, tensor: torch.Tensor, mean: float = 0, std: float = 0.09) -> torch.Tensor:
        """
        This function is used to generate the truncated_normal random variables, since there isn't this initiation in `Pytorch`
        :param tensor: tensor
        :param mean: sample mean
        :param std: sample std
        """
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def __init__(self):
        super().__init__()
        self.ts_corr = Lambda(ts_corr)
        self.ts_cov = Lambda(ts_cov)
        self.ts_stddev = Lambda(ts_stddev)
        self.ts_zscore = Lambda(ts_zscore)
        self.ts_return = Lambda(ts_return)
        self.ts_decaylinear = Lambda(ts_decaylinear)
        self.ts_mean_extract = Lambda(ts_mean_extract)
        self.BN = nn.BatchNorm2d(1, affine=True, track_running_stats=True)
        self.ts_mean_pool = Lambda(ts_mean_pool)
        self.ts_max = Lambda(ts_max)
        self.ts_min = Lambda(ts_min)
        self.Flatten = nn.Flatten(1, 3)
        self.linear1 = nn.Linear(702, 30)
        self.linear1.weight = self.truncated_normal_(nn.Parameter(torch.empty(30, 702), requires_grad=True))
        self.linear1.bias = self.truncated_normal_(nn.Parameter(torch.empty(30), requires_grad=True))
        self.linear2 = nn.Linear(30, 1)
        self.linear2.weight = self.truncated_normal_(nn.Parameter(torch.empty(1, 30), requires_grad=True))
        self.linear2.bias = self.truncated_normal_(nn.Parameter(torch.empty(1), requires_grad=True))
        self.dropout = nn.Dropout(p=0.5)
        self.weights = self.truncated_normal_(nn.Parameter(torch.empty(1), requires_grad=True))
        self.bias = self.truncated_normal_(nn.Parameter(torch.empty(1), requires_grad=True))
        # self.weights = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, xb):
        # extract layer
        # xb = xb.type(torch.float64)
        xb_1 = self.BN(self.ts_corr(xb))  # N, 1, 36, 3  ; BN: 2 learnable parameters
        xb_2 = self.BN(self.ts_cov(xb))  # N, 1, 36, 3
        xb_3 = self.BN(self.ts_stddev(xb))  # N, 1, 9, 3
        xb_4 = self.BN(self.ts_zscore(xb))  # N, 1, 9, 3
        xb_5 = self.BN(self.ts_return(xb))  # N, 1, 9, 3
        xb_6 = self.BN(self.ts_decaylinear(xb))  # N, 1, 9, 3
        xb_7 = self.BN(self.ts_mean_extract(xb))  # N, 1, 9, 3

        # pool layer
        xb = torch.cat([xb_1, xb_2, xb_3, xb_4, xb_5, xb_6, xb_7], 2)  # N, 1, 117, 3
        xb_8 = self.BN(self.ts_mean_pool(xb))  # N, 1, 117, 1
        xb_9 = self.BN(self.ts_max(xb))  # N, 1, 117, 1
        xb_10 = self.BN(self.ts_min(xb))  # N, 1, 117, 1

        # flatten layer
        xb = torch.cat([self.Flatten(xb), self.Flatten(xb_8),
                        self.Flatten(xb_9), self.Flatten(xb_10)], 1)  # N, 702

        # fully connected layer & hidden layer
        xb = self.linear1(xb)  # N, 30 ; linear1: 2 learnable parameters
        xb = self.dropout(xb)
        xb = f.relu(xb)

        # output layer
        xb = self.linear2(xb)  # N, 1 ; linear2: 2 learnable parameters
        xb = xb * self.weights + self.bias  # N, 1 ; 2 learnable parameters
        return xb.view(-1)

