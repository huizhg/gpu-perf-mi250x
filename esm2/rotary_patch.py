import torch
from transformers.models.esm import modeling_esm as me

ORIGINAL_FORWARD = me.RotaryEmbedding.forward

def _rotary_forward(self, q, k):
    """Same math as RotaryEmbedding.forward, without the Python-side cache."""
    L = k.shape[-2]
    t = torch.arange(L, device=k.device).type_as(self.inv_freq)
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    return (me.apply_rotary_pos_emb(q, cos, sin).to(dtype=q.dtype),
            me.apply_rotary_pos_emb(k, cos, sin).to(dtype=k.dtype))

def apply():
    me.RotaryEmbedding.forward = _rotary_forward