"""Train AutoBot on ScenarioNet datasets."""
import random

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from te_lrp.data.dataset import AutoBotDataset
from te_lrp.training.trainer import Trainer


@hydra.main(version_base=None, config_path='configs', config_name='config')
def train(cfg: DictConfig):
    """Build datasets and run training.

    Inputs: cfg (DictConfig): data, model, training, and checkpoint settings.
    Returns: None; writes logs and checkpoints."""
    # Seed generators before constructing datasets and the model.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    config = OmegaConf.merge(cfg.method, cfg)
    train_set = AutoBotDataset(config=config)
    val_set = AutoBotDataset(config=config, is_validation=True)
    train_loader = DataLoader(train_set, batch_size=config.method.train_batch_size, shuffle=True,
                              num_workers=config.load_num_workers, collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set, batch_size=config.method.eval_batch_size, shuffle=False,
                            num_workers=config.load_num_workers, collate_fn=val_set.collate_fn)
    trainer = Trainer(config=config)
    if config.ckpt_path is not None:
        trainer.load_checkpoint(checkpoint_path=config.ckpt_path)
    try:
        trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=config.method.max_epochs)
    finally:
        train_set.close_files()
        val_set.close_files()


if __name__ == '__main__':
    train()
