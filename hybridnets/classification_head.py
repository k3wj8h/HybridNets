import torch
from torch import nn

from hybridnets.network_blocks import SeparableConvBlock, MemoryEfficientSwish, Swish

class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList([SeparableConvBlock(in_channels, in_channels, norm=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)
            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)
            feats.append(feat)
        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()
        return feats
		

class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList([SeparableConvBlock(in_channels, in_channels, norm=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0,2,3,1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)
            feats.append(feat)

        return torch.cat(feats, dim=1)
