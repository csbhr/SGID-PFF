import torch
import torch.nn as nn


class Smooth_Loss(nn.Module):
    def __init__(self):
        super(Smooth_Loss, self).__init__()

    def forward(self, x):
        loss_smooth = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                      torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

        return loss_smooth
