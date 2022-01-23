import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Gradient_Loss(nn.Module):
    def __init__(self, device):
        super(Gradient_Loss, self).__init__()
        self.sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_X = torch.from_numpy(self.sobel_filter_X).float().to(device)
        self.sobel_filter_Y = torch.from_numpy(self.sobel_filter_Y).float().to(device)

    def forward(self, output, gt):
        b, c, h, w = output.size()

        output_X_c, output_Y_c = [], []
        gt_X_c, gt_Y_c = [], []
        for i in range(c):
            output_grad_X = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            output_grad_Y = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)
            gt_grad_X = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            gt_grad_Y = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)

            output_X_c.append(output_grad_X)
            output_Y_c.append(output_grad_Y)
            gt_X_c.append(gt_grad_X)
            gt_Y_c.append(gt_grad_Y)

        output_X = torch.cat(output_X_c, dim=1)
        output_Y = torch.cat(output_Y_c, dim=1)
        gt_X = torch.cat(gt_X_c, dim=1)
        gt_Y = torch.cat(gt_Y_c, dim=1)

        grad_loss = torch.mean(torch.abs(output_X - gt_X)) + torch.mean(torch.abs(output_Y - gt_Y))

        return grad_loss
