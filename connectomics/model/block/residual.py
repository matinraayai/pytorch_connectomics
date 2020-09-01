from abc import ABC

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .basic import *


# 1. Residual blocks
# implemented with 2D conv
class ResidualBlock2DC2(nn.Module, ABC):
    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(ResidualBlock2DC2, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act(in_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        self.act = get_layer_activation(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y

    # implemented with 3D conv


class ResidualBlock2D(nn.Module, ABC):
    """
    Residual Block 2D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """

    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(ResidualBlock2D, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 3, 3), padding=(0, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         norm_mode=norm_mode)
        self.act = get_layer_activation(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y


class ResidualBlock3D(nn.Module, ABC):
    """Residual Block 3D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """

    def __init__(self, in_planes, out_planes, projection=False, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(ResidualBlock3D, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode,
                            norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         norm_mode=norm_mode)
        self.act = get_layer_activation(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y


class BottleneckDilated2D(nn.Module, ABC):
    """Bottleneck Residual Block 2D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """

    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn',
                 act_mode='elu'):
        super(BottleneckDilated2D, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode,
                            act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3, 3), dilation=(dilate, dilate),
                            padding=(dilate, dilate), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        self.act = get_layer_activation(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y


class BottleneckDilated3D(nn.Module, ABC):
    """Bottleneck Residual Block 3D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """

    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn',
                 act_mode='elu'):
        super(BottleneckDilated3D, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0), norm_mode=norm_mode,
                            act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3, 3, 3),
                            dilation=(1, dilate, dilate), padding=(1, dilate, dilate), pad_mode=pad_mode,
                            norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0), norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0),
                                         norm_mode=norm_mode)
        self.act = get_layer_activation(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y
