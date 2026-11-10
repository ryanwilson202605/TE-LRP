"""TE-LRP operators following the paper and appendix. Backward tensors carry relevance."""
import torch
from torch.nn import functional as F

EPSILON = 1e-6


def stabilize(value):
    """Stabilize value (Tensor) with sign_+; return a Tensor of the same shape."""
    return value + EPSILON * torch.where(value >= 0, 1.0, -1.0)


class LinearRule(torch.autograd.Function):
    """Affine epsilon-rule; parameters receive no relevance."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        """Save affine inputs and outputs; take tensors and return the output Tensor."""
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, output)
        return output

    @staticmethod
    def backward(ctx, relevance):
        """Redistribute relevance (Tensor) to inputs; return relevance and None for parameters."""
        input, weight, output = ctx.saved_tensors
        return input * ((relevance / stabilize(value=output)) @ weight), None, None


class AddRule(torch.autograd.Function):
    """Broadcast-aware addition with epsilon-rule propagation."""

    @staticmethod
    def forward(ctx, input, other):
        """Save two input tensors and return their sum."""
        ctx.save_for_backward(input, other)
        return input + other

    @staticmethod
    def backward(ctx, relevance):
        """Redistribute relevance (Tensor) and undo broadcasting; return two tensors."""
        input, other = ctx.saved_tensors
        scaled = relevance / stabilize(value=input + other)
        return (input * scaled).sum_to_size(input.shape), (other * scaled).sum_to_size(other.shape)


def add(input, other):
    """Add input and other (Tensor) with relevance propagation; return Tensor."""
    return AddRule.apply(input, other)


class IdentityRule(torch.autograd.Function):
    """Preserve forward values and pass relevance through unchanged."""

    @staticmethod
    def forward(ctx, input, output):
        """Return output (Tensor) while retaining input (Tensor) as the relevance destination."""
        return output

    @staticmethod
    def backward(ctx, relevance):
        """Return incoming relevance (Tensor) for the input and None for the output."""
        return relevance, None


class AffineRule(torch.autograd.Function):
    """Epsilon-rule for the learned element-wise LayerNorm affine transform."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        """Transform input with weight and bias (Tensor); return the affine output Tensor."""
        contribution = input * weight
        output = contribution + bias
        ctx.save_for_backward(contribution, output)
        return output

    @staticmethod
    def backward(ctx, relevance):
        """Return relevance for the normalized input and None for affine parameters."""
        contribution, output = ctx.saved_tensors
        return relevance * contribution / stabilize(value=output), None, None


class MatmulRule(torch.autograd.Function):
    """Epsilon/uniform matrix-product rule, with key-only allocation for IIQ."""

    @staticmethod
    def forward(ctx, left, right, scale, iiq):
        """Multiply left and right (Tensor) with scale (float) and iiq (bool); return Tensor."""
        output = (left @ right) * scale
        ctx.save_for_backward(left, right, output)
        ctx.scale, ctx.iiq = scale, iiq
        return output

    @staticmethod
    def backward(ctx, relevance):
        """Redistribute relevance (Tensor) to the operands; return relevance tensors and None for controls."""
        left, right, output = ctx.saved_tensors
        scaled = relevance * ctx.scale / stabilize(value=output)
        right_relevance = right * (left.transpose(-2, -1) @ scaled)
        if ctx.iiq:
            return torch.zeros_like(left), right_relevance, None, None
        left_relevance = left * (scaled @ right.transpose(-2, -1))
        return left_relevance / 2, right_relevance / 2, None, None


class SoftmaxRule(torch.autograd.Function):
    """Propagate softmax relevance over valid keys only."""

    @staticmethod
    def forward(ctx, scores, valid):
        """Apply masked softmax to scores (Tensor) and valid (Boolean Tensor); return Tensor."""
        masked = scores.masked_fill(~valid, -torch.inf)
        output = torch.softmax(masked, dim=-1)
        ctx.save_for_backward(scores, output, valid)
        return output

    @staticmethod
    def backward(ctx, relevance):
        """Return score relevance (Tensor) and None for the mask."""
        scores, output, valid = ctx.saved_tensors
        relevance = relevance.masked_fill(~valid, 0)
        return (scores * (relevance - output * relevance.sum(dim=-1, keepdim=True))).masked_fill(~valid, 0), None
