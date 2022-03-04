from torch.nn import LayerNorm as LayerNorm
import torch
from DeepLayerNorm import DeepNorm

a = torch.rand((3,10))
# layer norm
ln = LayerNorm(10)
print(a)
y = ln(a)
print(y)

# deep norm
dn = DeepNorm(10)
d = dn(a)
print(d)