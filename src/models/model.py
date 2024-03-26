########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import torch
import torch.nn as nn

from src.models.resnet import NonBottleneck1D
from src.models.model_utils import ConvBNAct


class Decoder(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_decoder,
        activation=nn.ReLU(inplace=True),
        nr_decoder_blocks=1,
        encoder_decoder_fusion="add",
        upsampling_mode="bilinear",
        num_classes=37,
    ):
        super().__init__()

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes,
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes,
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes,
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels, num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode, channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode, channels=num_classes)

    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, _ = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, _ = self.decoder_module_2(out, enc_skip_down_8)
        out, _ = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        return out


class DecoderModule(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_dec,
        activation=nn.ReLU(inplace=True),
        nr_decoder_blocks=1,
        encoder_decoder_fusion="add",
        upsampling_mode="bilinear",
        num_classes=37,
    ):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(
            channels_in, channels_dec, kernel_size=3, activation=activation
        )

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(
                NonBottleneck1D(channels_dec, channels_dec, activation=activation)
            )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode, channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec, num_classes, kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == "add":
            out += encoder_features

        return out, out_side


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == "bilinear":
            self.align_corners = False
        else:
            self.align_corners = None

        if "learned-3x3" in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == "learned-3x3":
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(
                    channels, channels, groups=channels, kernel_size=3, padding=0
                )
            elif mode == "learned-3x3-zeropad":
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(
                    channels, channels, groups=channels, kernel_size=3, padding=1
                )

            # kernel that mimics bilinear interpolation
            w = torch.tensor(
                [
                    [
                        [
                            [0.0625, 0.1250, 0.0625],
                            [0.1250, 0.2500, 0.1250],
                            [0.0625, 0.1250, 0.0625],
                        ]
                    ]
                ]
            )

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = "nearest"
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.interp(x, size, mode=self.mode, align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x
