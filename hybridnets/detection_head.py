import torch
from torch import nn
import numpy as np
import itertools

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


class Anchors(nn.Module):
	def __init__(self, anchor_scale=4., pyramid_levels=[3, 4, 5, 6, 7], **kwargs):
		super().__init__()
		self.anchor_scale = anchor_scale
		self.pyramid_levels = pyramid_levels
		
		self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels]) # [8, 16, 32, 64, 128]
		self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])) # [1.		 1.62450479 2.4966611 ]
		self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]) # [(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)]
		self.last_anchors = {}
		self.last_shape = None

	def forward(self, image, dtype=torch.float32):
		image_shape = image.shape[2:]

		if image_shape == self.last_shape and image.device in self.last_anchors:
			return self.last_anchors[image.device]

		if self.last_shape is None or self.last_shape != image_shape:
			self.last_shape = image_shape

		boxes_all = []
		for stride in self.strides:
			boxes_level = []
			for scale, ratio in itertools.product(self.scales, self.ratios):
				#print(f'stride:{stride} - scale:{scale} - ratio:{ratio}')
				if image_shape[1] % stride != 0:
					raise ValueError('input size must be divided by the stride.')
				anchor_size_x_2 = self.anchor_scale * stride * scale * ratio[0] / 2.0
				anchor_size_y_2 = self.anchor_scale * stride * scale * ratio[1] / 2.0

				x = np.arange(stride / 2, image_shape[1], stride)
				y = np.arange(stride / 2, image_shape[0], stride)
				xv, yv = np.meshgrid(x, y)
				xv = xv.reshape(-1)
				yv = yv.reshape(-1)

				# y1,x1,y2,x2
				boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2, yv + anchor_size_y_2, xv + anchor_size_x_2))
				boxes = np.swapaxes(boxes, 0, 1)
				boxes_level.append(np.expand_dims(boxes, axis=1))
			# concat anchors on the same level to the reshape NxAx4
			boxes_level = np.concatenate(boxes_level, axis=1)
			boxes_all.append(boxes_level.reshape([-1, 4]))

		anchor_boxes = np.vstack(boxes_all)
		anchor_boxes = torch.from_numpy(anchor_boxes.astype(np.float32)).to(image.device)
		anchor_boxes = anchor_boxes.unsqueeze(0)

		# save it for later use to reduce overhead
		self.last_anchors[image.device] = anchor_boxes
		return anchor_boxes