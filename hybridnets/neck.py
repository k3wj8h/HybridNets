import torch
from torch import nn

from hybridnets.network_blocks import SeparableConvBlock, MaxPool2dStaticSamePadding, MemoryEfficientSwish, Swish, Conv2dStaticSamePadding


class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False):
        super(BiFPN,self).__init__()
        self.epsilon = epsilon
        self.first_time = first_time

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        
        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.p5_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.p6_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)
        self.p7_downsample = MaxPool2dStaticSamePadding(kernel_size=3, stride=2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        
        if self.first_time:
            self.p5_down_channel = nn.Sequential(Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                                                 nn.BatchNorm2d(num_channels, momentum=0.001, eps=1e-3))
            
            self.p4_down_channel = nn.Sequential(Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                                                 nn.BatchNorm2d(num_channels, momentum=0.001, eps=1e-3))
            
            self.p3_down_channel = nn.Sequential(Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                                                 nn.BatchNorm2d(num_channels, momentum=0.001, eps=1e-3))
            
            self.p5_to_p6 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                                          nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                                          MaxPool2dStaticSamePadding(kernel_size=3, stride=2))

            self.p6_to_p7 = nn.Sequential(MaxPool2dStaticSamePadding(kernel_size=3, stride=2))

            self.p4_down_channel_2 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                                                   nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3))

            self.p5_down_channel_2 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                                                   nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3))

        # Weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()


    def forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out
		