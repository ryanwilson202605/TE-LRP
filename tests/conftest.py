"""Small models and synthetic inputs for unit tests only."""
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def config(tmp_path):
    """Build a small CPU model configuration from tmp_path (Path); return DictConfig."""
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / 'te_lrp/configs/config.yaml')
    cfg.pop('defaults')
    cfg.pop('hydra')
    cfg.method = OmegaConf.load(root / 'te_lrp/configs/method/autobot.yaml')
    cfg.method.hidden_size = 16
    cfg.method.tx_hidden_size = 32
    cfg.method.tx_num_heads = 2
    cfg.method.num_encoder_layers = 1
    cfg.method.num_decoder_layers = 2
    cfg.method.max_num_agents = 3
    cfg.method.max_num_roads = 4
    cfg.method.max_points_per_lane = 3
    cfg.device = 'cpu'
    cfg.output_root = str(tmp_path)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return OmegaConf.merge(cfg.method, cfg)


@pytest.fixture
def batch(config):
    """Build valid and padded inputs from config (DictConfig); return a batch dict."""
    inputs = {
        'obj_trajs': torch.randn(2, 3, 21, 2),
        'obj_trajs_mask': torch.ones(2, 3, 21, dtype=torch.bool),
        'map_polylines': torch.randn(2, 4, 3, 2),
        'map_polylines_mask': torch.ones(2, 4, 3, dtype=torch.bool),
        'track_index_to_predict': torch.zeros(2, dtype=torch.long),
        'center_gt_trajs': torch.randn(2, 60, 4),
        'center_gt_trajs_mask': torch.ones(2, 60, dtype=torch.bool),
        'center_gt_final_valid_idx': torch.full((2,), 59, dtype=torch.long),
    }
    inputs['obj_trajs_mask'][:, -1] = False
    inputs['obj_trajs'][:, -1] = 0
    inputs['map_polylines_mask'][:, -1] = False
    inputs['map_polylines'][:, -1] = 0
    return {'batch_size': 2, 'input_dict': inputs}
