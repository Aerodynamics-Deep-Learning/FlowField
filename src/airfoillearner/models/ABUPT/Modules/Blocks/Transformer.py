import torch.nn as nn
from modules.mlp import Mlp

class Transformer(nn.Module):
  def __init__(self, dim: int, num_heads: int, attention):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.attn  = attention(dim, num_heads)
    self.norm2 = nn.LayerNorm(dim)
    self.mlp   = Mlp(dim)

  def forward(self, x):
      x = x + self.attn(self.norm1(x))
      x = x + self.mlp(self.norm2(x))
      return x
