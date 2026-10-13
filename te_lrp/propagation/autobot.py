"""Enable TE-LRP rules on a single AutoBot instance."""
from types import MethodType

from torch import nn

from te_lrp.propagation.layers import attention_forward, decoder_forward, encoder_forward, layer_norm_forward, linear_forward, relu_forward
from te_lrp.propagation.rules import add
from te_lrp.models.autobot import AutoBotEgo, MapEncoderPts, PositionalEncoding


def enable_te_lrp(model):
    """Enable paper propagation rules on one AutoBot instance.

    Inputs: model (AutoBotEgo): model with loaded weights.
    Returns: AutoBotEgo: the same instance in evaluation mode with frozen parameters."""
    model.eval()
    model.requires_grad_(False)
    forwards = {
        nn.Linear: linear_forward,
        nn.ReLU: relu_forward,
        nn.LayerNorm: layer_norm_forward,
        nn.MultiheadAttention: attention_forward,
        nn.TransformerEncoderLayer: encoder_forward,
        nn.TransformerDecoderLayer: decoder_forward,
    }
    for module in model.modules():
        if type(module) in forwards:
            module.forward = MethodType(forwards[type(module)], module)
        if isinstance(module, nn.MultiheadAttention):
            module.te_lrp_iiq = False
        if isinstance(module, (AutoBotEgo, MapEncoderPts, PositionalEncoding)):
            module.add = add
    # P is a fixed query; the subsequent map query is scene-dependent.
    model.prob_decoder.te_lrp_iiq = True
    return model
