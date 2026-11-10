"""Cross-mode similarity of jointly normalized entity attributions."""
import torch


def attribution_similarity(entity_distribution, distribution_valid):
    """Compute base-2 Jensen-Shannon similarity and the mean over distinct pairs.

    Inputs: entity_distribution (Tensor[B,K,E]): normalized entity distributions;
        distribution_valid (Tensor[B,K]): flags for positive attribution totals.
    Returns: dict[str, Tensor]: pairwise similarities and validity [B,K,K],
        scene means and validity [B]. Undefined results are NaN.
    """
    # Float64 reduces cancellation when two distributions are nearly identical.
    probabilities = entity_distribution.to(torch.float64)
    left = probabilities.unsqueeze(2)
    right = probabilities.unsqueeze(1)
    mixture = (left + right) / 2
    # Zero-probability terms contribute zero, including jointly padded entities.
    left_term = torch.where(left > 0, left * (left.log2() - mixture.log2()), 0.)
    right_term = torch.where(right > 0, right * (right.log2() - mixture.log2()), 0.)
    divergence = (left_term + right_term).sum(dim=-1) / 2
    pairwise_valid = distribution_valid.unsqueeze(2) & distribution_valid.unsqueeze(1)
    pairwise = (1 - divergence.clamp(0, 1).sqrt()).to(entity_distribution.dtype)
    pairwise = pairwise.masked_fill(~pairwise_valid, torch.nan)
    num_modes = probabilities.shape[1]
    pairs = torch.triu_indices(num_modes, num_modes, offset=1, device=probabilities.device)
    scene_valid = distribution_valid.all(dim=-1) & (num_modes > 1)
    scene = pairwise[:, pairs[0], pairs[1]].mean(dim=-1)
    scene = scene.masked_fill(~scene_valid, torch.nan)
    return {
        'pairwise_similarity': pairwise,
        'pairwise_similarity_valid': pairwise_valid,
        'scene_similarity': scene,
        'scene_similarity_valid': scene_valid,
    }
