"""Local PyTorch layers for the CTSM decoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AttentionProjections(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = head_dim

        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, self.q_size + 2 * self.kv_size)
        self.o_proj = nn.Linear(self.q_size, hidden_size)

        self.scaling = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32))

    def _scale_queries(self, query: torch.Tensor) -> torch.Tensor:
        feature_scale = F.softplus(self.scaling) * (1.442695041 / math.sqrt(self.head_dim))
        return query * feature_scale


class TransformerMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.layer_norm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        paddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = F.relu(self.gate_proj(self.layer_norm(x)))
        branch = self.down_proj(hidden)

        if paddings is not None:
            branch = branch * (1.0 - paddings[..., None])

        return branch + x


class ResidualBlock(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(input_dims, hidden_dims), nn.SiLU())
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden_layer(x)
        return self.output_layer(hidden) + self.residual_layer(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        inverse_rms = torch.rsqrt((x32 ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * inverse_rms * self.weight.float()).to(dtype=x.dtype)
