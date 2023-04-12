import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class conv_block(nn.Module):
    def __init__(self, inputD, outputD, activation, threshold):
        super(conv_block, self).__init__()

        if activation == 'tilu':
            self.activation = utils.TilU(threshold)
        elif activation == 'silu':
            self.activation = utils.SiLU()
        else:
            self.activation = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inputD, outputD, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(outputD)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(outputD, outputD, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(outputD)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)

        return x

class up_conv(nn.Module):

    def __init__(self, inputD, outputD, activation, threshold):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(inputD, outputD, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(outputD)
        )

        if activation == 'tilu':
            self.activation = utils.TilU(threshold)
        elif activation == 'silu':
            self.activation = utils.SiLU()
        else:
            self.activation = nn.ReLU()
    def forward(self, x):
        x = self.up(x)
        x = self.activation(x)
        return x

class Attention_block(nn.Module):

    def __init__(self, d_input, e_input, output, f_d):
        super(Attention_block, self).__init__()

        self.W_e = nn.Sequential(
            nn.Conv2d(e_input, output, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(output)
        )

        self.W_d = nn.Sequential(
            nn.Conv2d(d_input, output, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(output)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(output, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.fl = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(f_d * f_d, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace = True)

    def forward(self, D, E):

        d = self.W_d(D)
        e = self.W_e(E)

        fmap = self.relu(d * e)
        psi = self.psi(fmap)

        att = torch.where(psi > 0.5, 1, 0.2)
        psi = self.fl(psi)
        predict = self.fc(psi)


        out = E * att
        return out, predict

    

