import torch
from torch import nn

import math


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
		

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
		
		
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm: x = self.bn(x)
        
        return x
		
		
class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = [self.pool.stride] * 2
        self.kernel_size = [self.pool.kernel_size] * 2
    
    def forward(self, x):
        h, w = x.shape[-2:]
            
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = nn.functional.pad(x, [left, right, top, bottom])
        x = self.pool(x)
        return x
		
		
class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
          self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
          self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
          self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
          self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w, = x.shape[-2:]

        extra_h = (math.ceil(w/self.stride[1])-1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h/self.stride[0])-1) * self.stride[0] - h + self.kernel_size[0]

        left, top = extra_h//2, extra_v//2
        right, bottom = extra_h-left, extra_v-top

        x = nn.functional.pad(x, [left, right, top, bottom])
        x = self.conv(x)
        return x
		
		
class Conv3x3BNSwish(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.swish = Swish()
        self.upsample = upsample
        self.block = nn.Sequential(Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3))
        self.conv_sp = SeparableConvBlock(out_channels, onnx_export=False)

    def forward(self, x):
        x = self.conv_sp(self.swish(self.block(x)))
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x
		
		
class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [Conv3x3BNSwish(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3BNSwish(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)
		
		
class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, x):
        return sum(x)