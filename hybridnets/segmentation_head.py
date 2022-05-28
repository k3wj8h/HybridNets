from torch import nn

from hybridnets.network_blocks import SegmentationBlock, MergeBlock


class BiFPNDecoder(nn.Module):
    def __init__(self, encoder_depth=5, pyramid_channels=64, segmentation_channels=64, dropout=0.2, merge_policy="add"):
        super().__init__()
        self.seg_blocks = nn.ModuleList([SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples) for n_upsamples in [5, 4, 3, 2, 1]])
        self.seg_p2 = SegmentationBlock(32, 64, n_upsamples=0)
        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, inputs):
        p2, p3, p4, p5, p6, p7 = inputs
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p7, p6, p5, p4, p3])]  
        p2 = self.seg_p2(p2)      
        p3,p4,p5,p6,p7 = feature_pyramid
        x = self.merge((p2,p3,p4,p5,p6,p7))
        x = self.dropout(x)
        return x
		

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Softmax(dim=1)
        super().__init__(conv2d, upsampling, activation)
