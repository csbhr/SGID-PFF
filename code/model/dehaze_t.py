import torch.nn as nn
import torch
from model import trans
from loss import smooth_loss
import model.blocks as blocks


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    return DEHAZE_T(img_channels=args.n_colors, t_channels=args.t_channels, n_resblock=args.n_resblock,
                    n_feat=args.n_feat, device=device)


class FusionModule(nn.Module):
    def __init__(self, n_feat, kernel_size=5):
        super(FusionModule, self).__init__()
        print("Creating BRB-Fusion-Module")
        self.block1 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)
        self.block2 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)

    def forward(self, x, y):
        H_0 = x + y

        x_1, y_1, H_1 = self.block1(x, y, H_0)
        x_2, y_2, H_2 = self.block2(x_1, y_1, H_1)

        return H_2


class DEHAZE_T(nn.Module):

    def __init__(self, img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device='cuda'):
        super(DEHAZE_T, self).__init__()
        print("Creating Dehaze-T Net")
        self.device = device

        self.extra_feat = nn.Sequential(
            nn.Conv2d(img_channels, n_feat, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            blocks.ResBlock(n_feat, n_feat, kernel_size=5, stride=1)
        )
        self.fusion_feat = FusionModule(n_feat=n_feat, kernel_size=5)
        self.trans_net = trans.TRANS(in_channels=1, out_channels=t_channels,
                                     n_resblock=n_resblock, n_feat=n_feat, feat_in=True)
        self.smooth_loss = smooth_loss.Smooth_Loss()

    def forward(self, x, pre_est_J):
        b, c, h, w = x.size()

        x_feat = self.extra_feat(x)
        pre_est_J_feat = self.extra_feat(pre_est_J)

        fusioned_feat = self.fusion_feat(x_feat, pre_est_J_feat)

        trans, _ = self.trans_net(fusioned_feat)
        air = torch.ones(b, 1, h, w).to(self.device)

        output = (1 / trans) * x + ((trans - 1) / trans) * air

        mid_loss = self.smooth_loss(trans)

        return output, trans, air, mid_loss
