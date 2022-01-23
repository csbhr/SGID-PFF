import torch.nn as nn
import torch
from model.pre_dehaze_t import PRE_DEHAZE_T
from model.dehaze_t import DEHAZE_T


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    pretrain_pre_dehaze_pt = args.pretrain_models_dir + 'pretrain_pre_dehaze_net.pt' if not args.test_only else '.'
    return DEHAZE_SGID_PFF(img_channels=args.n_colors, t_channels=args.t_channels, n_resblock=args.n_resblock,
                           n_feat=args.n_feat, pretrain_pre_dehaze_pt=pretrain_pre_dehaze_pt, device=device)


class DEHAZE_SGID_PFF(nn.Module):

    def __init__(self, img_channels=3, t_channels=1, n_resblock=3, n_feat=32,
                 pretrain_pre_dehaze_pt='.', device='cuda'):
        super(DEHAZE_SGID_PFF, self).__init__()
        print("Creating Dehaze-SGID-PFF Net")
        self.device = device

        self.pre_dehaze = PRE_DEHAZE_T(img_channels=img_channels, t_channels=t_channels, n_resblock=n_resblock,
                                       n_feat=n_feat, device=device)
        self.dehaze = DEHAZE_T(img_channels=img_channels, t_channels=t_channels, n_resblock=n_resblock,
                               n_feat=n_feat, device=device)

        if pretrain_pre_dehaze_pt != '.':
            self.pre_dehaze.load_state_dict(torch.load(pretrain_pre_dehaze_pt))
            print('Loading pre dehaze model from {}'.format(pretrain_pre_dehaze_pt))

    def forward(self, x):
        pre_est_J, _, _, _ = self.pre_dehaze(x)

        output, trans, air, mid_loss = self.dehaze(x, pre_est_J)

        return pre_est_J, output, trans, air, mid_loss
