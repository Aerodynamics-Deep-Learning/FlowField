import torch
import torch.nn as nn

# Initially this embedding was written as a normal function. However, this approach was problematic when moving the model to GPU. Since as a function, this is not moved when 'model.to('mps')' is used and causes(aciklamayi sonra tamamla)
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        dimPerAxis = dim // ndim
        dimPerTrigFunc = dimPerAxis // 2
        idx = torch.arange(dimPerTrigFunc, dtype=torch.float32)
        idx = 1.0 / (max_wavelength ** (2.0 * idx / dimPerAxis))
        self.register_buffer("idx", idx)
        self.padding = dim - (dimPerTrigFunc * 2 * ndim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        angles = coords.unsqueeze(-1) * self.idx
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        embedding = embedding.flatten(start_dim=-2)
        if self.padding > 0:
            pad_shape = (*embedding.shape[:-1], self.padding)
            embedding = torch.cat([embedding, embedding.new_zeros(pad_shape)], dim=-1)
        return embedding