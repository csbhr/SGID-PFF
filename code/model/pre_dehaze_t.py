import torch.nn as nn
import torch
from model import trans
from loss import smooth_loss


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    return PRE_DEHAZE_T(img_channels=args.n_colors, t_channels=args.t_channels, n_resblock=args.n_resblock,
                        n_feat=args.n_feat, device=device)


class PRE_DEHAZE_T(nn.Module):

    def __init__(self, img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device='cuda'):
        super(PRE_DEHAZE_T, self).__init__()
        print("Creating Pre-Dehaze-T Net")
        self.device = device

        self.trans_net = trans.TRANS(in_channels=img_channels, out_channels=t_channels,
                                     n_resblock=n_resblock, n_feat=n_feat)
        self.smooth_loss = smooth_loss.Smooth_Loss()

    def forward(self, x):
        b, c, h, w = x.size()

        trans, _ = self.trans_net(x)
        air = torch.ones(b, 1, h, w).to(self.device)

        output = (1 / trans) * x + ((trans - 1) / trans) * air

        mid_loss = self.smooth_loss(trans)

        return output, trans, air, mid_loss
