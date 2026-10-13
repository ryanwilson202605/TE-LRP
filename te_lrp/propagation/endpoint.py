"""Coordinate-wise endpoint attribution, entity aggregation, and similarity."""
import torch

from te_lrp.similarity import attribution_similarity


def attribute(model, input_dict):
    """Attribute all x/y prediction endpoints using one shared forward pass.

    Inputs: model (AutoBotEgo): TE-LRP-enabled model; input_dict (dict): model inputs.
    Returns: dict[str, Tensor]: predictions, input relevance, entity vectors, magnitudes, distributions, and similarity scores."""
    if model.training:
        raise ValueError('TE-LRP requires evaluation mode')
    # Create attribution leaves and hold other inputs fixed.
    inputs = {}
    for key, mask_key in [('obj_trajs', 'obj_trajs_mask'), ('map_polylines', 'map_polylines_mask')]:
        value = input_dict[key].detach().clone()
        value = value.masked_fill(~input_dict[mask_key].unsqueeze(-1), 0)
        inputs[key] = value.requires_grad_(True)
    static = {key: value.detach() if isinstance(value, torch.Tensor) else value
              for key, value in input_dict.items() if key not in inputs}
    agents, maps = inputs['obj_trajs'], inputs['map_polylines']
    prediction = model.predict(batch={'input_dict': {**static, **inputs}})
    endpoints = prediction['predicted_trajectory'][:, :, -1, :2]
    num_modes = endpoints.shape[1]
    relevance = [[], []]
    # Seed each backward pass with the signed predicted coordinate.
    for mode in range(num_modes):
        coordinates = [[], []]
        for coordinate in range(2):
            target = endpoints[:, mode, coordinate]
            values = torch.autograd.grad(target, (agents, maps), grad_outputs=target.detach(),
                                         retain_graph=mode != num_modes - 1 or coordinate != 1)
            for index, value in enumerate(values):
                coordinates[index].append(value.detach())
        for index in range(2):
            relevance[index].append(torch.stack(coordinates[index], dim=1))
    agent_relevance, map_relevance = [torch.stack(values, dim=1) for values in relevance]
    agent_relevance = agent_relevance.masked_fill(~static['obj_trajs_mask'][:, None, None, :, :, None], 0)
    map_relevance = map_relevance.masked_fill(~static['map_polylines_mask'][:, None, None, :, :, None], 0)

    # Sum signed relevance before computing entity vector norms.
    agent_vectors = agent_relevance.sum(dim=(-1, -2)).transpose(-1, -2)
    map_vectors = map_relevance.sum(dim=(-1, -2)).transpose(-1, -2)
    entity_vectors = torch.cat([agent_vectors, map_vectors], dim=2)
    entity_valid = torch.cat([static['obj_trajs_mask'].any(dim=-1), static['map_polylines_mask'].any(dim=-1)], dim=1)
    magnitude = torch.linalg.vector_norm(entity_vectors, dim=-1)
    totals = magnitude.sum(dim=-1, keepdim=True)
    distribution_valid = totals.squeeze(-1) > 0
    distribution = torch.where(distribution_valid.unsqueeze(-1), magnitude / totals.clamp_min(torch.finfo(totals.dtype).tiny), torch.nan)
    if not torch.isfinite(entity_vectors).all():
        raise FloatingPointError('TE-LRP relevance is non-finite; check inputs and weights')
    return {
        'predicted_trajectory': prediction['predicted_trajectory'].detach(),
        'predicted_probability': prediction['predicted_probability'].detach(),
        'agent_relevance': agent_relevance,
        'map_relevance': map_relevance,
        'entity_vectors': entity_vectors,
        'entity_magnitude': magnitude,
        'entity_distribution': distribution,
        'entity_valid': entity_valid,
        'distribution_valid': distribution_valid,
        **attribution_similarity(entity_distribution=distribution, distribution_valid=distribution_valid),
    }
