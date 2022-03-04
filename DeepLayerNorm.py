import torch
from torch import Tensor, Size
from typing import Union, List, Tuple
from torch.nn import LayerNorm as LayerNorm

_shape_t = Union[int, List[int], Size]
NUBER_LAYER = 1000 # Encoder/Decoder

class DeepNorm(torch.nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(DeepNorm, self).__init__()

        self.alpha = (2 * NUBER_LAYER) ** 0.25
        self.layernorm = LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


    def forward(self, x):
        x_normed = self.layernorm(x)
        return self.alpha * x + x_normed