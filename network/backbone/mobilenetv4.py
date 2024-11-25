from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_config import MODEL_SPECS 
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
    ) -> int:
    """
    This function is copied from here 
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    
    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_excitation:
            self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)
        
class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, 
            inp, 
            oup, 
            start_dw_kernel_size, 
            middle_dw_kernel_size, 
            middle_dw_downsample,
            stride,
            expand_ratio
        ):
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:            
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        
        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)
        
    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x

def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ =  ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


# class MobileNetV4(nn.Module):
#     def __init__(self, model):
#         # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
#         # MobileNetV4HybridMedium  MobileNetV4HybridLarge
#         """Params to initiate MobilenNetV4
#         Args:
#             model : support 5 types of models as indicated in 
#             "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"        
#         """
#         super().__init__()
#         assert model in MODEL_SPECS.keys()
#         self.model = model
#         self.spec = MODEL_SPECS[self.model]
       
#         # conv0
#         self.conv0 = build_blocks(self.spec['conv0'])
#         # layer1
#         self.layer1 = build_blocks(self.spec['layer1'])
#         # layer2
#         self.layer2 = build_blocks(self.spec['layer2'])
#         # layer3
#         self.layer3 = build_blocks(self.spec['layer3'])
#         # layer4
#         self.layer4 = build_blocks(self.spec['layer4'])
#         # layer5   
#         self.layer5 = build_blocks(self.spec['layer5'])
#         self.features = nn.ModuleList([self.conv0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])     
#         self.channels = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
        
#     def forward(self, x):
#         input_size = x.size(2)
#         scale = [4, 8, 16, 32]
#         features = [None, None, None, None]
#         for f in self.features:
#             x = f(x)
#             if input_size // x.size(2) in scale:
#                 features[scale.index(input_size // x.size(2))] = x
#         return features


class MobileNetV4(nn.Module):
    def __init__(self, model):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
        """Params to initiate MobilenNetV4
        Args:
            model : support 5 types of models as indicated in 
            "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"        
        """
        super().__init__()
        assert model in MODEL_SPECS.keys()
        self.model = model
        self.spec = MODEL_SPECS[self.model]
       
        # conv0
        self.conv0 = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1 = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2 = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3 = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4 = build_blocks(self.spec['layer4'])
        # layer5   
        self.layer5 = build_blocks(self.spec['layer5'])
        self.features = nn.ModuleList([self.conv0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5])    
        self.channels = [i.size(1) for i in self.warmup(torch.randn(1, 3, 640, 640))]
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(self.channels, self.out_channels)
        self.combined_features_inplanes = self.forward((torch.randn(1, 3, 640, 640)))['out'].size(1)

    def warmup(self, x):
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]
        for f in self.features:
            x = f(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x
        return features
    
    def forward(self, x):
        input_size = x.size(2)
        scale = [4, 8, 16, 32]
        features = [None, None, None, None]
        for f in self.features:
            x = f(x)
            if input_size // x.size(2) in scale:
                features[scale.index(input_size // x.size(2))] = x
        
        fpn_features = OrderedDict()
        fpn_features_names = ['0', '1', '2', '3']
        for i in range(4):
            fpn_features[fpn_features_names[i]] = features[i]
        
        fpn_features = self.fpn(fpn_features)
        max_size = max([f.size(2) for f in features])
        features_upsampled = [F.interpolate(f, size=(max_size, max_size), mode='bilinear', align_corners=False) for f in features]
        combined_features = OrderedDict()

        combined_features['out'] = torch.cat(features_upsampled, dim=1)

        return combined_features
    
    # def forward(self, x):
    #     input_size = x.size(2)
    #     scale = [4, 8, 16, 32]
    #     features = [None, None, None, None]
    #     for f in self.features:
    #         x = f(x)
    #         if input_size // x.size(2) in scale:
    #             features[scale.index(input_size // x.size(2))] = x
    #     return features




