import torch.nn as nn
import model.blocks as blocks


def make_model(args):
    return UNet(in_channels=args.n_colors, out_channels=args.n_colors, n_resblock=args.n_resblock, n_feat=args.n_feat)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_resblock=3, n_feat=32, kernel_size=5, feat_in=False):
        super(UNet, self).__init__()
        print("Creating U-Net")

        InBlock = []
        if not feat_in:
            InBlock.extend([nn.Sequential(
                nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            )])
        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1) for _ in range(n_resblock)])

        # encoder1
        Encoder_1 = [nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_1.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)])
        # encoder2
        Encoder_2 = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_2.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)])

        # decoder2
        Decoder_2 = [blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                     for _ in range(n_resblock)]
        Decoder_2.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))
        # decoder1
        Decoder_1 = [blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                     for _ in range(n_resblock)]
        Decoder_1.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))

        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1) for _ in range(n_resblock)]
        OutBlock.extend([
            nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        ])

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_1 = nn.Sequential(*Encoder_1)
        self.encoder_2 = nn.Sequential(*Encoder_2)
        self.decoder_2 = nn.Sequential(*Decoder_2)
        self.decoder_1 = nn.Sequential(*Decoder_1)
        self.outBlock = nn.Sequential(*OutBlock)

    def forward(self, x):
        inblock_out = self.inBlock(x)
        encoder_1_out = self.encoder_1(inblock_out)
        encoder_2_out = self.encoder_2(encoder_1_out)
        decoder_2_out = self.decoder_2(encoder_2_out)
        decoder_1_out = self.decoder_1(decoder_2_out + encoder_1_out)
        out = self.outBlock(decoder_1_out + inblock_out)

        mid_loss = None

        return out, mid_loss
