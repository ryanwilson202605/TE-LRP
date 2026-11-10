"""Single-device AutoBot training, validation, and checkpoint management."""
from pathlib import Path

import torch
from tqdm import tqdm

from te_lrp.models.autobot import AutoBotEgo
from te_lrp.training.logger import Logger
from te_lrp.training.metrics import compute_metrics


class Trainer:
    """Manage AutoBot training and checkpoint restoration from a DictConfig."""

    def __init__(self, config):
        """Initialize the model, optimizer, and logger from config (DictConfig); return None."""
        self.config = config
        self.device = torch.device(config.device)
        self.model = AutoBotEgo(config=config).to(self.device)
        optimizers, schedulers = self.model.configure_optimizers()
        self.optimizer, self.scheduler = optimizers[0], schedulers[0]
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.logger = Logger(log_dir=str(Path(config.log_dir)), exp_name=config.exp_name)
        self.logger.log_config(config=config)

    def save_checkpoint(self, is_best):
        """Save resumable state and optionally the best checkpoint.

        Inputs: is_best (bool): whether the validation metric improved.
        Returns: None; writes latest_ckpt and optionally best_ckpt."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        paths = [Path(self.config.latest_ckpt)]
        if is_best:
            paths.append(Path(self.config.best_ckpt))
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        """Restore training state from checkpoint_path (str or Path); return None."""
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']

    def run_epoch(self, loader, training):
        """Run a training or validation epoch and average metrics by sample count.

        Inputs: loader (DataLoader), training (bool).
        Returns: dict[str, float]: loss and, during validation, prediction metrics."""
        self.model.train(training)
        totals, count = {}, 0
        for batch in tqdm(loader, desc='train' if training else 'val'):
            batch['input_dict'] = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch['input_dict'].items()
            }
            with torch.set_grad_enabled(training):
                prediction, loss = self.model(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('AutoBot loss is non-finite; stopping before saving invalid weights')
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.method.grad_clip_norm)
                    self.optimizer.step()
                    self.global_step += 1
                metrics = {'loss': loss.item()}
                if not training:
                    metrics.update(compute_metrics(batch=batch, prediction=prediction))
            size = batch['batch_size']
            count += size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
        if count == 0:
            raise ValueError('The data loader contains no samples')
        return {key: value / count for key, value in totals.items()}

    def fit(self, train_loader, val_loader, max_epochs):
        """Train, validate, and save checkpoints for the requested epoch range.

        Inputs: train_loader, val_loader (DataLoader), max_epochs (int).
        Returns: None; writes metrics and checkpoints."""
        for epoch in range(self.current_epoch, max_epochs):
            train_metrics = self.run_epoch(loader=train_loader, training=True)
            val_metrics = self.run_epoch(loader=val_loader, training=False)
            self.scheduler.step()
            self.current_epoch = epoch + 1
            metrics = {f'train_{key}': value for key, value in train_metrics.items()}
            metrics.update({f'val_{key}': value for key, value in val_metrics.items()})
            self.logger.log_metrics(metrics=metrics, epoch=self.current_epoch)
            is_best = val_metrics['brier_fde'] < self.best_metric
            self.best_metric = min(self.best_metric, val_metrics['brier_fde'])
            self.save_checkpoint(is_best=is_best)
            print(f'Epoch {self.current_epoch}: {metrics}')
