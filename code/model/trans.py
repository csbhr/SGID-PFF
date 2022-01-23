import torch
import torch.nn as nn
import model.unet as unet


def make_model(args):
    return TRANS(in_channels=args.n_colors, out_channels=args.t_channels, n_resblock=args.n_resblock, n_feat=args.n_feat)


class TRANS(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, n_resblock=3, n_feat=32, feat_in=False):
        super(TRANS, self).__init__()
        print("Creating Trans Net")

        self.unet_body = unet.UNet(in_channels=in_channels, out_channels=out_channels,
                                   n_resblock=n_resblock, n_feat=n_feat, feat_in=feat_in)
        self.sigmoid = nn.Sigmoid()

    def clamp_trans(self, trans):
        # trans<0 => 1e-8
        mask = (trans.detach() > 0).float()
        little = torch.ones_like(trans) * 1e-8
        trans = trans * mask + little * (1 - mask)
        # trans>1 => (1 - 1e-8)
        mask = (trans.detach() < 1).float()
        large = torch.ones_like(trans) * (1 - 1e-8)
        trans = trans * mask + large * (1 - mask)
        return trans

    def forward(self, x):
        unet_out, _ = self.unet_body(x)
        est_trans = self.sigmoid(unet_out)
        clamped_est_trans = self.clamp_trans(est_trans)

        mid_loss = None

        return clamped_est_trans, mid_loss
