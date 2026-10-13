"""Test ScenarioNet PKL loading and caching using synthetic scenes."""
import pickle

import numpy as np
import pytest

from te_lrp.data.dataset import AutoBotDataset


def write_database(path, dataset_name, interval=0.1):
    """Write a minimal ScenarioNet database.

    Inputs: path (Path), dataset_name (str), interval (float).
    Returns: Path: the generated scene PKL."""
    path.mkdir()
    time = np.arange(81, dtype=np.float32)
    state = {'position': np.column_stack([time, time * 0, time * 0]),
             'length': np.full(81, 4.), 'width': np.full(81, 2.), 'height': np.full(81, 1.5),
             'heading': time * 0, 'velocity': np.column_stack([time * 0 + 10, time * 0]),
             'valid': np.ones(81)}
    scenario = {'tracks': {'ego': {'type': 'VEHICLE', 'state': state}},
                'map_features': {'lane': {'type': 'LANE_SURFACE_STREET',
                                         'polyline': np.array([[0., 0., 0.], [40., 0., 0.], [80., 0., 0.]])}},
                'metadata': {'ts': time * interval, 'sdc_id': 'ego', 'scenario_id': 'same_id',
                             'dataset': dataset_name}}
    filename = f'sd_{dataset_name}_v1_same_id.pkl'
    for name, value in [(filename, scenario), ('dataset_summary.pkl', {filename: {}}),
                        ('dataset_mapping.pkl', {filename: ''})]:
        with (path / name).open('wb') as stream:
            pickle.dump(value, stream)
    return path / filename


def test_first_cache_reload_and_source_isolation(config, tmp_path):
    """Check cache creation, reuse, source isolation, and unchanged input files.

    Inputs: config (DictConfig), tmp_path (Path): test fixtures.
    Returns: None."""
    a, b = tmp_path / 'a', tmp_path / 'b'
    source = write_database(path=a, dataset_name='av2')
    write_database(path=b, dataset_name='waymo')
    original = source.read_bytes()
    config.val_data_path = [str(a), str(b)]
    dataset = AutoBotDataset(config=config, is_validation=True)
    assert len(dataset) == 2
    record = dataset[0]
    assert record['agent_ids'][record['track_index_to_predict']] == 'ego'
    assert dataset[1]['dataset_name'] == 'waymo'
    assert record['obj_trajs'].shape[0:2] == (3, 21)
    assert source.read_bytes() == original
    cache_files = list((tmp_path / 'cache').rglob('samples.h5'))
    mtimes = {path: path.stat().st_mtime_ns for path in cache_files}
    dataset.close_files()
    reloaded = AutoBotDataset(config=config, is_validation=True)
    np.testing.assert_array_equal(record['obj_trajs'], reloaded[0]['obj_trajs'])
    assert mtimes == {path: path.stat().st_mtime_ns for path in cache_files}
    reloaded.close_files()


def test_bad_sampling_interval_fails(config, tmp_path):
    """Check that an unsupported sampling interval raises; takes fixtures and returns None."""
    source = tmp_path / 'bad'
    write_database(path=source, dataset_name='av2', interval=0.2)
    config.val_data_path = [str(source)]
    with pytest.raises(ValueError, match='sampling interval'):
        AutoBotDataset(config=config, is_validation=True)
