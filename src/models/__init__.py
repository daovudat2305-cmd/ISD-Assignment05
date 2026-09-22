"""
Model definitions for ISD-Assignment05.
"""

from .basic_cnn import build_basic_cnn
from .lenet import build_lenet5
from .alexnet import build_alexnet
from .vgg import build_vgg_custom, build_vgg16_transfer
from .resnet import build_resnet_custom, build_resnet50_transfer
from .tabular_cnn import build_1d_cnn, build_mlp

__all__ = [
    "build_basic_cnn",
    "build_lenet5",
    "build_alexnet",
    "build_vgg_custom",
    "build_vgg16_transfer",
    "build_resnet_custom",
    "build_resnet50_transfer",
    "build_1d_cnn",
    "build_mlp",
]
