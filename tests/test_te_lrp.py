"""Test TE-LRP rules, prediction equivalence, and endpoint attribution."""
from copy import deepcopy

import pytest
import torch

from te_lrp.propagation import enable_te_lrp
from te_lrp.propagation.rules import AddRule, IdentityRule, LinearRule, MatmulRule, SoftmaxRule, stabilize
from te_lrp.models.autobot import AutoBotEgo
from te_lrp.propagation.endpoint import attribute


def test_affine_and_add_rules():
    """Check bias, negative values, and epsilon against hand calculations; returns None."""
    x = torch.tensor([[2., -1.]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([[3., 4.]], dtype=torch.float64)
    output = LinearRule.apply(x, weight, torch.tensor([-3.], dtype=torch.float64))
    relevance = torch.autograd.grad(output, x, grad_outputs=torch.tensor([[5.]], dtype=torch.float64))[0]
    torch.testing.assert_close(relevance, torch.tensor([[6., -4.]], dtype=torch.float64) * 5 / (-1 - 1e-6))
    a = torch.tensor([1., -1.], requires_grad=True)
    b = torch.tensor([[-1., 2.], [2., -2.]], requires_grad=True)
    y = AddRule.apply(a, b)
    ra, rb = torch.autograd.grad(y, (a, b), grad_outputs=torch.ones_like(y))
    expected = 1 / stabilize(value=a + b)
    torch.testing.assert_close(ra, (a * expected).sum(0))
    torch.testing.assert_close(rb, b * expected)


def test_identity_preserves_relevance_at_zero():
    """Check identity propagation at zero input; returns None."""
    x = torch.tensor([0., -2.], requires_grad=True)
    y = IdentityRule.apply(x, torch.tensor([3., 0.]))
    seed = torch.tensor([-4., 5.])
    torch.testing.assert_close(torch.autograd.grad(y, x, grad_outputs=seed)[0], seed)


def test_iiq_and_standard_matmul():
    """Check zero query relevance for IIQ and equal splitting for ordinary products; returns None."""
    q = torch.tensor([[2., 3.]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[4.], [5.]], dtype=torch.float64, requires_grad=True)
    seed = torch.tensor([[7.]], dtype=torch.float64)
    standard = MatmulRule.apply(q, k, 0.5, False)
    rq, rk = torch.autograd.grad(standard, (q, k), grad_outputs=seed)
    iiq = MatmulRule.apply(q, k, 0.5, True)
    iq, ik = torch.autograd.grad(iiq, (q, k), grad_outputs=seed)
    assert torch.equal(iq, torch.zeros_like(iq))
    expected = torch.tensor([[4.], [7.5]], dtype=torch.float64) * 7 / (11.5 + 1e-6)
    torch.testing.assert_close(ik, expected)
    torch.testing.assert_close(ik, rk * 2)
    torch.testing.assert_close(rq.sum(), rk.sum())


def test_softmax_valid_support_and_empty_rows():
    """Check masked softmax rules and fully masked rows; returns None."""
    scores = torch.tensor([[1., 2., 100.], [1., 2., 3.]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [False, False, False]])
    probabilities = SoftmaxRule.apply(scores, valid)
    seed = torch.tensor([[3., 4., 1000.], [5., 6., 7.]])
    relevance = torch.autograd.grad(probabilities, scores, grad_outputs=seed)[0]
    expected = scores[0, :2] * (seed[0, :2] - scores[0, :2].softmax(0) * 7)
    torch.testing.assert_close(relevance[0, :2], expected)
    assert relevance[0, 2] == 0
    reference = torch.softmax(scores.masked_fill(~valid, -torch.inf), dim=-1)
    torch.testing.assert_close(probabilities, reference, equal_nan=True)
    assert torch.isnan(probabilities[1]).all()
    assert (relevance[1] == 0).all()


def test_model_forward_and_instance_isolation(config, batch):
    """Check predictions, instance isolation, aggregation, and independence from ground truth.

    Inputs: config (DictConfig), batch (dict): test fixtures.
    Returns: None."""
    model = AutoBotEgo(config=config).eval()
    other = deepcopy(model)
    with torch.no_grad():
        reference = model.predict(batch=batch)
    enable_te_lrp(model=model)
    patched = model.predict(batch=batch)
    for key in reference:
        torch.testing.assert_close(reference[key], patched[key], atol=2e-5, rtol=2e-5)
    assert not hasattr(other.agents_dynamic_encoder[0], 'forward') or 'forward' not in other.agents_dynamic_encoder[0].__dict__
    # Changing or removing ground truth must not affect attribution.
    without_gt = {key: value for key, value in batch['input_dict'].items() if not key.startswith('center_gt')}
    result = attribute(model=model, input_dict=without_gt)
    repeated = attribute(model=model, input_dict=batch['input_dict'])
    for key in result:
        torch.testing.assert_close(result[key], repeated[key], equal_nan=True)
    assert result['agent_relevance'].shape == (2, 6, 2, 3, 21, 2)
    assert result['map_relevance'].shape == (2, 6, 2, 4, 3, 2)
    assert (result['agent_relevance'][:, :, :, -1] == 0).all()
    assert (result['map_relevance'][:, :, :, -1] == 0).all()
    torch.testing.assert_close(result['entity_vectors'][:, :, :3], result['agent_relevance'].sum((-1, -2)).transpose(-1, -2))
    torch.testing.assert_close(result['entity_distribution'].sum(-1), torch.ones(2, 6))


def test_all_map_invalid_preserves_native_forward(config, batch):
    """Preserve native predictions for empty maps and reject non-finite attribution.

    Inputs: config (DictConfig), batch (dict): small-model fixtures.
    Returns: None."""
    # Keep one ordinary sample and fully mask the second sample's map.
    batch['input_dict']['map_polylines'][1].zero_()
    batch['input_dict']['map_polylines_mask'][1].zero_()
    model = AutoBotEgo(config=config).eval()
    with torch.no_grad():
        reference = model.predict(batch=batch)
    enable_te_lrp(model=model)
    with torch.no_grad():
        actual = model.predict(batch=batch)
    for key in reference:
        assert torch.isfinite(reference[key][0]).all()
        assert not torch.isfinite(reference[key][1]).all()
        torch.testing.assert_close(actual[key], reference[key], atol=2e-5, rtol=2e-5, equal_nan=True)
    # Reject non-finite attribution before saving the batch.
    with pytest.raises(FloatingPointError, match='relevance is non-finite'):
        attribute(model=model, input_dict=batch['input_dict'])


def test_iiq_only_first_map_query(config, batch):
    """Check IIQ boundaries across two decoder layers; takes config and batch fixtures, returns None."""
    model = enable_te_lrp(model=AutoBotEgo(config=config))
    flags = []
    handle = model.map_attn_layers.register_forward_pre_hook(lambda module, args: flags.append(module.te_lrp_iiq))
    model.predict(batch=batch)
    handle.remove()
    assert flags == [True, False]
    assert model.prob_decoder.te_lrp_iiq
    assert not model.mode_map_attn.te_lrp_iiq
    assert not model.map_encoder.road_pts_attn_layer.te_lrp_iiq


def test_zero_endpoint_distribution(config, batch):
    """Check zero relevance and undefined distributions for zero endpoints; takes fixtures, returns None."""
    model = AutoBotEgo(config=config)
    with torch.no_grad():
        model.output_model.observation_model[-1].weight[:2].zero_()
        model.output_model.observation_model[-1].bias[:2].zero_()
    result = attribute(model=enable_te_lrp(model=model), input_dict=batch['input_dict'])
    assert (result['entity_vectors'] == 0).all()
    assert not result['distribution_valid'].any()
    assert result['entity_distribution'].isnan().all()


class EndpointExample(torch.nn.Module):
    """A two-input endpoint model for hand-checking attribution initialization."""

    def predict(self, batch):
        """Return hand-computable endpoints and uniform probabilities from batch (dict)."""
        inputs = batch['input_dict']
        values = torch.cat([inputs['obj_trajs'].flatten(1), inputs['map_polylines'].flatten(1)], dim=1)
        weight = values.new_tensor([[1., 2.], [3., -1.]])
        endpoint = LinearRule.apply(values, weight, None)
        trajectory = endpoint[:, None, None, :].expand(-1, 6, 60, -1)
        return {'predicted_trajectory': trajectory, 'predicted_probability': values.new_full((1, 6), 1 / 6)}


def test_signed_coordinate_initialization():
    """Check signed coordinate seeds and entity L2 aggregation; returns None."""
    inputs = {'obj_trajs': torch.tensor([[[[1.]]]]), 'map_polylines': torch.tensor([[[[-2.]]]]),
              'obj_trajs_mask': torch.ones(1, 1, 1, dtype=torch.bool),
              'map_polylines_mask': torch.ones(1, 1, 1, dtype=torch.bool)}
    result = attribute(model=EndpointExample().eval(), input_dict=inputs)
    expected = torch.tensor([[1., 3.], [-4., 2.]])
    expected[:, 0] *= -3 / (-3 - 1e-6)
    expected[:, 1] *= 5 / (5 + 1e-6)
    torch.testing.assert_close(result['entity_vectors'][0, 0], expected)
    torch.testing.assert_close(result['entity_magnitude'][0, 0], expected.norm(dim=-1))
