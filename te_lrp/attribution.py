"""Save AutoBot predictions, TE-LRP attributions, and cross-mode similarity."""
from pathlib import Path
from urllib.parse import quote

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from te_lrp.propagation import enable_te_lrp
from te_lrp.models.autobot import AutoBotEgo
from te_lrp.propagation.endpoint import attribute
from te_lrp.data.dataset import AutoBotDataset


def save_results(results, input_dict, save_dir):
    """Save one NPZ per target with input identifiers.

    Inputs: results (dict[str, Tensor]), input_dict (dict), save_dir (Path).
    Returns: None; writes prediction, attribution, and similarity files."""
    results_np = {key: value.detach().cpu().numpy() for key, value in results.items()}
    for index in range(results_np['predicted_trajectory'].shape[0]):
        dataset = str(input_dict['dataset_name'][index])
        scenario = str(input_dict['scenario_id'][index])
        target = str(input_dict['center_objects_id'][index])
        scene_dir = save_dir / quote(dataset, safe='').replace('.', '%2E') / quote(scenario, safe='').replace('.', '%2E')
        scene_dir.mkdir(parents=True, exist_ok=True)
        output = {key: value[index] for key, value in results_np.items()}
        for key in ['obj_trajs', 'obj_trajs_mask', 'map_polylines', 'map_polylines_mask',
                    'track_index_to_predict', 'center_objects_world', 'agent_ids']:
            value = input_dict[key][index]
            output[key] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=str)
        output.update(dataset_name=dataset, scenario_id=scenario, target_id=target)
        output['map_entity_index'] = np.arange(output['map_polylines'].shape[0])
        np.savez_compressed(scene_dir / f'{quote(target, safe="")}.npz', **output)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig):
    """Run endpoint attribution on the configured validation sources.

    Inputs: cfg (DictConfig): checkpoint, dataset, device, and output settings.
    Returns: None; saves settings, predictions, attributions, and similarities."""
    OmegaConf.set_struct(cfg, False)
    config = OmegaConf.merge(cfg.method, cfg)
    device = torch.device(config.device)
    model = AutoBotEgo(config=config).to(device)
    checkpoint = torch.load(Path(config.eval_ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    enable_te_lrp(model=model)
    dataset = AutoBotDataset(config=config, is_validation=True)
    loader = DataLoader(dataset, batch_size=config.attribution.batch_size, shuffle=False,
                        num_workers=config.load_num_workers, collate_fn=dataset.collate_fn)
    save_dir = Path(config.attribution.attribution_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=save_dir / 'config.yaml', resolve=True)
    try:
        for batch_index, batch in enumerate(tqdm(loader, desc='TE-LRP')):
            if batch_index < config.attribution.batch_start:
                continue
            limit = config.attribution.batch_limit
            if limit is not None and batch_index >= config.attribution.batch_start + limit:
                break
            input_dict = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                          for key, value in batch['input_dict'].items()}
            with torch.enable_grad():
                results = attribute(model=model, input_dict=input_dict)
            save_results(results=results, input_dict=input_dict, save_dir=save_dir)
    finally:
        dataset.close_files()


if __name__ == '__main__':
    main()
