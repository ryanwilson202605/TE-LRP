"""Trajectory metrics for AutoBot validation."""
import numpy as np
import torch


def compute_metrics(batch, prediction):
    """Compute minADE, minFDE, miss rate, and brier_fde over valid time steps.

    Inputs: batch (dict): ground truth and masks; prediction (dict[str, Tensor]).
    Returns: dict[str, float]: batch-averaged validation metrics."""
    inputs = batch['input_dict']
    gt_traj = inputs['center_gt_trajs'].unsqueeze(1)
    gt_traj_mask = inputs['center_gt_trajs_mask'].unsqueeze(1)
    center_gt_final_valid_idx = inputs['center_gt_final_valid_idx']

    predicted_traj = prediction['predicted_trajectory']
    predicted_prob = prediction['predicted_probability'].detach().cpu().numpy()

    # Average displacement errors over valid time steps.
    ade_diff = torch.norm(predicted_traj[:, :, :, :2] - gt_traj[:, :, :, :2], 2, dim=-1)
    ade_losses = torch.sum(ade_diff * gt_traj_mask, dim=-1) / torch.sum(gt_traj_mask, dim=-1)
    ade_losses = ade_losses.cpu().detach().numpy()
    minade = np.min(ade_losses, axis=1)

    # Evaluate FDE at each sample's last valid time step.
    bs, modes, future_len = ade_diff.shape
    center_gt_final_valid_idx = center_gt_final_valid_idx.view(-1, 1, 1).repeat(1, modes, 1).to(torch.int64)

    fde = torch.gather(ade_diff, -1, center_gt_final_valid_idx).cpu().detach().numpy().squeeze(-1)
    minfde = np.min(fde, axis=-1)

    # Use the best-FDE mode probability for brier_fde.
    best_fde_idx = np.argmin(fde, axis=-1)
    predicted_prob = predicted_prob[np.arange(bs), best_fde_idx]
    miss_rate = (minfde > 2.0)
    brier_fde = minfde + np.square(1 - predicted_prob)

    loss_dict = {
        'minADE6': np.mean(minade),
        'minFDE6': np.mean(minfde),
        'miss_rate': np.mean(miss_rate.astype(np.float32)),
        'brier_fde': np.mean(brier_fde)
    }

    return loss_dict
