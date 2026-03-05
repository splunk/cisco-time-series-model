from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from timesfm import pytorch_patched_decoder as ppd


class RotaryPositionalEmbedding(nn.Module):
  """Rotary positional embedding (RoPE) helper for attention.

  This is parameter-free (buffers only) and rotates query/key vectors based on position.
  """

  def __init__(self, dim: int, base: float = 10_000.0):
    super().__init__()
    if dim % 2 != 0:
      raise ValueError(f"RoPE requires an even head_dim, got dim={dim}.")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids.ndim == 1:
      position_ids = position_ids.unsqueeze(0)
    
    pos = position_ids.to(torch.float32)
    freqs = torch.einsum("bt,d->btd", pos, self.inv_freq)
    
    return freqs.cos().to(dtype), freqs.sin().to(dtype)

  def apply(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [B, T, D/2]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    
    return torch.cat([first_part, second_part], dim=-1)


class CTSMAttentionRoPE(ppd.TimesFMAttention):
  """Cisco TSM attention with rotary positional embeddings applied to Q/K."""

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rope_base: float = 10_000.0,
  ):
    super().__init__(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    self.rope = RotaryPositionalEmbedding(head_dim, base=rope_base)

  def forward(
      self,
      hidden_states: torch.Tensor,
      mask: torch.Tensor,
      kv_write_indices: Optional[torch.Tensor] = None,
      kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      position_ids: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if kv_cache is not None or kv_write_indices is not None:
      raise NotImplementedError("RoPE attention currently supports non-cached forward only.")

    hidden_states_shape = hidden_states.shape
    assert len(hidden_states_shape) == 3

    B, input_len, _ = hidden_states_shape

    qkv = self.qkv_proj(hidden_states)
    xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    xq = xq.view(B, -1, self.num_heads, self.head_dim)
    xk = xk.view(B, -1, self.num_kv_heads, self.head_dim)
    xv = xv.view(B, -1, self.num_kv_heads, self.head_dim)
    xq = self._per_dim_scaling(xq)

    key = xk
    value = xv
    if self.num_kv_heads != self.num_heads:
      key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
      value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

    # [B, n_local_heads, input_len, head_dim]
    q = xq.transpose(1, 2)
    # [B, n_local_heads, input_len, head_dim]
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    seq_len = k.shape[2]
    if position_ids is None:
      position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
      position_ids = position_ids.unsqueeze(0).expand(B, -1)
    elif position_ids.ndim == 1:
      position_ids = position_ids.unsqueeze(0).expand(B, -1)

    if position_ids.shape[1] != seq_len:
      raise ValueError(f"position_ids length {position_ids.shape[1]} != seq_len {seq_len}")

    cos, sin = self.rope.cos_sin(position_ids, dtype=q.dtype)
    q = self.rope.apply(q, cos, sin)
    k = self.rope.apply(k, cos, sin)

    # [B, n_local_heads, input_len, max_seq_len]
    scores = torch.matmul(q, k.transpose(2, 3))
    scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(q)

    # [B, n_local_heads, input_len, head_dim]
    output = torch.matmul(scores, v)

    # [B, input_len, hidden_dim]
    output = output.transpose(1, 2).contiguous().view(B, input_len, -1)
    output = self.o_proj(output)
    return scores, output


class CTSMDecoderLayerRoPE(ppd.TimesFMDecoderLayer):
  """Cisco TSM decoder layer with RoPE attention."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      rms_norm_eps: float = 1e-6,
      rope_base: float = 10_000.0,
  ):
    super().__init__(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
    )
    self.self_attn = CTSMAttentionRoPE(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_base=rope_base,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      mask: torch.Tensor,
      paddings: torch.Tensor,
      kv_write_indices: Optional[torch.Tensor] = None,
      kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      position_ids: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if position_ids is None:
      seq_len = hidden_states.shape[1]
      position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
      position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    scores, hidden_states = self.self_attn(
        hidden_states=hidden_states,
        mask=mask,
        kv_write_indices=kv_write_indices,
        kv_cache=kv_cache,
        position_ids=position_ids,
    )
    hidden_states = residual + hidden_states
    hidden_states = self.mlp(hidden_states, paddings=paddings)
    return scores, hidden_states


class StackedDecoderRoPE(nn.Module):
  """Stacked decoder with RoPE-enabled layers."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      num_layers: int,
      rms_norm_eps: float = 1e-6,
      rope_base: float = 10_000.0,
  ):
    super().__init__()
    self.layers = nn.ModuleList([
        CTSMDecoderLayerRoPE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            rope_base=rope_base,
        ) for _ in range(num_layers)
    ])

  def forward(
      self,
      hidden_states: torch.Tensor,
      paddings: torch.Tensor,
      kv_write_indices: Optional[torch.Tensor] = None,
      kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
      position_ids: Optional[torch.Tensor] = None,
      mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if mask is None:
      padding_mask = ppd.convert_paddings_to_mask(paddings, hidden_states.dtype)
      atten_mask = ppd.causal_mask(hidden_states)
      mask = ppd.merge_masks(padding_mask, atten_mask)

    if position_ids is None:
      seq_len = hidden_states.shape[1]
      position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
      position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

    for i in range(len(self.layers)):
      layer = self.layers[i]
      kv_cache = kv_caches[i] if kv_caches is not None else None
      _, hidden_states = layer(
          hidden_states=hidden_states,
          mask=mask,
          paddings=paddings,
          position_ids=position_ids,
          kv_write_indices=kv_write_indices,
          kv_cache=kv_cache,
      )
    return hidden_states
