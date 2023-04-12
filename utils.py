
import torch.nn as nn # import modules
import torch

def tilu(input, threshold):

    return torch.where(input > threshold, input, 0)

def silu(input):
    return input * torch.sigmoid(input)

class TilU(nn.Module):
    """
    Input shape = Output shape
    tilu(x) = {x, if x > threshold
               0, otherwise}
    """
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return tilu(x, self.threshold)


class SiLU(nn.Module):
    """
    Input shape = Output shape
    silu(x) = x * sigmoid(x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return silu(x)

def init_weights(m):

    if type(m) == nn.Conv2d:
            # get the number of the inputs
        n = m.in_features
        y = 0.5/np.sqrt(n)
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        m.bias.data.fill_(0)

