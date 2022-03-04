import torch
from torch import Tensor, Size
from typing import Union, List, Tuple
from torch.nn import LayerNorm as LayerNorm

_shape_t = Union[int, List[int], Size]
NUMBER_LAYER = 1000 # Encoder/Decoder

class DeepNorm(torch.nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True):
        """
            Deep Layer Normalization
        :param normalized_shape: input shape from an expected input of size
        :param eps:  a value added to the denominator for numerical stability, default 1e-8
        :param elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        """
        super(DeepNorm, self).__init__()

        self.alpha = (2 * NUMBER_LAYER) ** 0.25
        self.layernorm = LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)


    def forward(self, x):
        x_normed = self.layernorm(x)
        return self.alpha * x + x_normed
