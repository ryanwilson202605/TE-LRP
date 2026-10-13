"""Save training configurations and epoch metrics."""
import csv
from pathlib import Path

from omegaconf import OmegaConf


class Logger:
    """Write resolved training configuration and epoch metrics."""

    def __init__(self, log_dir, exp_name):
        """Create a log directory from log_dir and exp_name (str); return None."""
        self.log_path = Path(log_dir) / exp_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_path / 'metrics.csv'

    def log_config(self, config):
        """Save config (DictConfig) with resolved interpolations; return None."""
        OmegaConf.save(config=config, f=self.log_path / 'config.yaml', resolve=True)

    def log_metrics(self, metrics, epoch):
        """Append metrics (dict[str, float]) and epoch (int) to CSV; return None."""
        row = {'epoch': epoch, **metrics}
        exists = self.metrics_file.exists()
        with self.metrics_file.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
