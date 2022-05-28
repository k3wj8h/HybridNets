import torch
from torch import nn

from hybridnets.neck import BiFPN
from hybridnets.detection_head import Classifier, Regressor
from hybridnets.segmentation_head import BiFPNDecoder, SegmentationHead

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_model_params, url_map
import numpy as np
import itertools


class HNBackBone(nn.Module):
	def __init__(self, num_classes, seg_classes=1, **kwargs):
		super(HNBackBone, self).__init__()
		self.seg_classes = seg_classes
		self.num_classes = num_classes

		self.fpn_num_filters = 160
		self.fpn_cell_repeats = 6
		self.input_sizes = 896
		self.box_class_repeats = 4
		self.pyramid_levels = 5
		self.anchor_scale = 1.25
		self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
		self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
		conv_channel_coef = [48, 136, 384] # the channels of P3/P4/P5

		num_anchors = len(self.aspect_ratios) * self.num_scales

		self.bifpn = nn.Sequential(*[BiFPN(self.fpn_num_filters, conv_channel_coef, first_time=True if _==0 else False) for _ in range(self.fpn_cell_repeats)])

		self.regressor = Regressor(in_channels=self.fpn_num_filters, num_anchors=num_anchors, num_layers=self.box_class_repeats, pyramid_levels=self.pyramid_levels)
		
		self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters)

		self.segmentation_head = SegmentationHead(in_channels=64, out_channels=self.seg_classes+1, kernel_size=1, upsampling=4)
		
		self.classifier = Classifier(in_channels=self.fpn_num_filters, num_anchors=num_anchors, num_classes=num_classes, num_layers=self.box_class_repeats, pyramid_levels=self.pyramid_levels)

		self.anchors = Anchors(anchor_scale=self.anchor_scale, pyramid_levels=(torch.arange(self.pyramid_levels) + 3).tolist(), **kwargs)
		
		# EfficientNet_Pytorch - default params of efficientnet-b3
		self.encoder = EfficientNetEncoder(stage_idxs=(5,8,18,26), out_channels=(3,40,32,48,136,384), model_name="efficientnet-b3", depth=5)
		self.encoder.load_state_dict(torch.utils.model_zoo.load_url(url_map["efficientnet-b3"]))

		self.initialize_decoder(self.bifpndecoder)
		self.initialize_head(self.segmentation_head)
		self.initialize_decoder(self.bifpn)
		

	def forward(self, inputs):
		max_size = inputs.shape[-1]
		p2, p3, p4, p5 = self.encoder(inputs)[-4:]
		features = (p3, p4, p5)
		features = self.bifpn(features)
		p3, p4, p5, p6, p7 = features
		outputs = self.bifpndecoder((p2, p3, p4, p5, p6, p7))
		segmentation = self.segmentation_head(outputs)
		regression = self.regressor(features) # [10, 46035, 4]
		classification = self.classifier(features) # [10, 46035, 4]
		anchors = self.anchors(inputs, inputs.dtype) # [1, 46035, 4]
		return features, regression, classification, anchors, segmentation

	def initialize_decoder(self, module):
		for m in module.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def initialize_head(self, module):
		for m in module.modules():
			if isinstance(m, (nn.Linear, nn.Conv2d)):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
					

class EfficientNetEncoder(EfficientNet):
	def __init__(self, stage_idxs, out_channels, model_name, depth=5):
		blocks_args, global_params = get_model_params(model_name, override_params=None)
		super().__init__(blocks_args, global_params)

		self._stage_idxs = stage_idxs
		self._out_channels = out_channels
		self._depth = depth
		self._in_channels = 3

		del self._fc

	def get_stages(self):
		return [
			nn.Identity(),
			nn.Sequential(self._conv_stem, self._bn0, self._swish),
			self._blocks[:self._stage_idxs[0]],
			self._blocks[self._stage_idxs[0]:self._stage_idxs[1]],
			self._blocks[self._stage_idxs[1]:self._stage_idxs[2]],
			self._blocks[self._stage_idxs[2]:],
		]

	def forward(self, x):
		stages = self.get_stages()

		block_number = 0.
		drop_connect_rate = self._global_params.drop_connect_rate

		features = []
		for i in range(self._depth + 1):

			# Identity and Sequential stages
			if i < 2:
				x = stages[i](x)

			# Block stages need drop_connect rate
			else:
				for module in stages[i]:
					drop_connect = drop_connect_rate * block_number / len(self._blocks)
					block_number += 1.
					x = module(x, drop_connect)

			features.append(x)

		return features

	def load_state_dict(self, state_dict, **kwargs):
		state_dict.pop("_fc.bias", None)
		state_dict.pop("_fc.weight", None)
		super().load_state_dict(state_dict, **kwargs)
		
		
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