# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import *
from .resnet_encoder import ResnetEncoder


class DepthDecoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=True, scales=range(4), 
                    num_output_channels=1, use_skips=True, alpha=10, beta=0.01):
        super(DepthDecoder, self).__init__()

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=1)
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.alpha = alpha
        self.beta = beta

        self.num_ch_enc = self.encoder.num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs["upconv_{}_{}".format(i,0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs["upconv_{}_{}".format(i,1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs["dispconv_{}".format(s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for lay in self.convs.values():
            for m in lay.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, img):
        input_features = self.encoder(img)

        self.outputs = []

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs["upconv_{}_{}".format(i,0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs["upconv_{}_{}".format(i,1)](x)
            if i in self.scales:
                self.outputs.append(self.alpha * self.sigmoid(self.convs["dispconv_{}".format(i)](x)) + self.beta) 

        if self.training:
            return self.outputs[::-1]
        else:
            return self.outputs[-1]

if __name__ == '__main__':
    model = DepthDecoder()
    x = torch.rand(1,3,128,416)
    for i in model(x):
        print(i.shape)