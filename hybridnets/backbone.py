import torch
from torch import nn

from hybridnets.neck import BiFPN
from hybridnets.detection_head import Classifier, Regressor, Anchors
from hybridnets.segmentation_head import BiFPNDecoder, SegmentationHead

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_model_params, url_map


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
