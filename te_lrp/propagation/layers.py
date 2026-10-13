"""Instance-level PyTorch layer forwards for TE-LRP."""
import torch
from torch.nn import functional as F

from .rules import AddRule, AffineRule, IdentityRule, LinearRule, MatmulRule, SoftmaxRule


def linear_forward(self, input):
    """Map input (Tensor) through the linear layer with epsilon-rule backward; return Tensor."""
    return LinearRule.apply(input, self.weight, self.bias)


def relu_forward(self, input):
    """Apply ReLU to input (Tensor) with identity relevance backward; return Tensor."""
    return IdentityRule.apply(input, F.relu(input))


def layer_norm_forward(self, input):
    """Normalize input (Tensor) with identity relevance and affine epsilon propagation; return Tensor."""
    normalized = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
    normalized = IdentityRule.apply(input, normalized)
    return AffineRule.apply(normalized, self.weight, self.bias)


def attention_forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                      attn_mask=None, average_attn_weights=True, is_causal=False):
    """Compute multihead attention with explicit relevance propagation.

    Inputs: query, key, value (Tensor[L,B,D]); optional masks (Tensor); Boolean attention options.
    Returns: tuple[Tensor, Tensor | None]: output and optional attention weights."""
    # Reuse the original Q/K/V projection parameters.
    weights = self.in_proj_weight.chunk(3)
    biases = self.in_proj_bias.chunk(3)
    query = LinearRule.apply(query, weights[0], biases[0])
    key = LinearRule.apply(key, weights[1], biases[1])
    value = LinearRule.apply(value, weights[2], biases[2])
    target_len, batch_size, embed_dim = query.shape
    source_len, head_dim = key.shape[0], embed_dim // self.num_heads
    query = query.reshape(target_len, batch_size, self.num_heads, head_dim).permute(1, 2, 0, 3)
    key = key.reshape(source_len, batch_size, self.num_heads, head_dim).permute(1, 2, 0, 3)
    value = value.reshape(source_len, batch_size, self.num_heads, head_dim).permute(1, 2, 0, 3)
    scores = MatmulRule.apply(query, key.transpose(-2, -1), head_dim ** -0.5, self.te_lrp_iiq)

    # Masks receive no relevance; fully masked rows retain native non-finite outputs.
    valid = torch.ones_like(scores, dtype=torch.bool)
    if key_padding_mask is not None:
        blocked = key_padding_mask if key_padding_mask.dtype == torch.bool else torch.isneginf(key_padding_mask)
        valid = valid & ~blocked[:, None, None, :]
    if attn_mask is not None:
        blocked = attn_mask if attn_mask.dtype == torch.bool else torch.isneginf(attn_mask)
        valid = valid & ~blocked
    elif is_causal:
        valid = valid & torch.ones(target_len, source_len, device=query.device, dtype=torch.bool).tril()
    attention = SoftmaxRule.apply(scores, valid)
    output = MatmulRule.apply(attention, value, 1.0, False)
    output = output.permute(2, 0, 1, 3).reshape(target_len, batch_size, embed_dim)
    output = self.out_proj(output)
    weights = attention.mean(dim=1) if average_attn_weights else attention
    return output, weights if need_weights else None


def encoder_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    """Run an AutoBot post-norm encoder with relevance-aware operators.

    Inputs: src (Tensor), optional masks (Tensor), is_causal (bool).
    Returns: Tensor: encoder output with evaluation-time prediction semantics."""
    attention = self.self_attn(src, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask, need_weights=False,
                               is_causal=is_causal)[0]
    src = self.norm1(AddRule.apply(src, attention))
    hidden = self.linear1(src)
    hidden = IdentityRule.apply(hidden, self.activation(hidden))
    return self.norm2(AddRule.apply(src, self.linear2(hidden)))


def decoder_forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    tgt_is_causal=False, memory_is_causal=False):
    """Run an AutoBot post-norm decoder with relevance-aware operators.

    Inputs: tgt, memory (Tensor), optional masks (Tensor), causal flags (bool).
    Returns: Tensor: decoder output with evaluation-time prediction semantics."""
    attention = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask, need_weights=False,
                               is_causal=tgt_is_causal)[0]
    tgt = self.norm1(AddRule.apply(tgt, attention))
    attention = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask, need_weights=False,
                                    is_causal=memory_is_causal)[0]
    tgt = self.norm2(AddRule.apply(tgt, attention))
    hidden = self.linear1(tgt)
    hidden = IdentityRule.apply(hidden, self.activation(hidden))
    return self.norm3(AddRule.apply(tgt, self.linear2(hidden)))
