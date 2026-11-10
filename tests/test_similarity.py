"""Check paper-defined similarity against analytical and SciPy references."""
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

from te_lrp.attribution import save_results
from te_lrp.models.autobot import AutoBotEgo
from te_lrp.propagation import enable_te_lrp
from te_lrp.propagation.endpoint import attribute
from te_lrp.similarity import attribution_similarity


def test_similarity_reference_and_padding():
    """Check sparse distributions, symmetry, bounds, padding, and distinct-pair means."""
    probabilities = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [.5, .5, 0.]],
                                  [[.2, .3, .5], [.2, .3, .5], [.8, .1, .1]]], dtype=torch.float64)
    valid = torch.ones(2, 3, dtype=torch.bool)
    result = attribution_similarity(entity_distribution=probabilities, distribution_valid=valid)
    expected = np.array([[[1 - jensenshannon(p, q, base=2) for q in scene]
                          for p in scene] for scene in probabilities.numpy()])
    np.testing.assert_allclose(result['pairwise_similarity'], expected, atol=1e-12)
    np.testing.assert_allclose(result['scene_similarity'], expected[:, [0, 0, 1], [1, 2, 2]].mean(-1))
    assert result['pairwise_similarity'][0, 0, 1] == 0
    assert result['pairwise_similarity'][1, 0, 1] == 1
    padded = attribution_similarity(entity_distribution=torch.nn.functional.pad(probabilities, (0, 4)),
                                   distribution_valid=valid)
    for key in result:
        torch.testing.assert_close(result[key], padded[key])


def test_invalid_mode_and_single_mode():
    """Keep valid pairs while marking incomplete scene scores and single-mode means undefined."""
    probabilities = torch.tensor([[[1., 0.], [1., 0.], [torch.nan, torch.nan]]])
    result = attribution_similarity(entity_distribution=probabilities,
                                    distribution_valid=torch.tensor([[True, True, False]]))
    assert (result['pairwise_similarity'][0, :2, :2] == 1).all()
    assert result['pairwise_similarity'][0, 2].isnan().all()
    assert result['pairwise_similarity'][0, :, 2].isnan().all()
    assert not result['pairwise_similarity_valid'][0, 2].any()
    assert result['scene_similarity'].isnan().all()
    assert not result['scene_similarity_valid'].any()
    single = attribution_similarity(entity_distribution=probabilities[:, :1],
                                    distribution_valid=torch.ones(1, 1, dtype=torch.bool))
    assert single['pairwise_similarity'].item() == 1
    assert single['scene_similarity'].isnan().all()
    assert not single['scene_similarity_valid'].any()


def test_attribution_similarity_npz(config, batch, tmp_path):
    """Verify model attribution integrates similarity and saves readable per-target scores.

    Inputs: config (DictConfig), batch (dict), tmp_path (Path): synthetic fixtures.
    Returns: None.
    """
    result = attribute(model=enable_te_lrp(model=AutoBotEgo(config=config)), input_dict=batch['input_dict'])
    assert result['pairwise_similarity'].shape == (2, 6, 6)
    assert result['scene_similarity_valid'].all()
    expected = np.array([1 - jensenshannon(result['entity_distribution'][0, i].double().numpy(),
                                          result['entity_distribution'][0, j].double().numpy(), base=2)
                         for i in range(6) for j in range(i + 1, 6)])
    np.testing.assert_allclose(result['scene_similarity'][0], expected.mean(), atol=1e-6)
    inputs = dict(batch['input_dict'], dataset_name=['synthetic'] * 2, scenario_id=['a', 'b'],
                  center_objects_id=['ego'] * 2, center_objects_world=torch.zeros(2, 10),
                  agent_ids=np.array([['ego', 'other', '']] * 2))
    save_results(results=result, input_dict=inputs, save_dir=tmp_path)
    with np.load(tmp_path / 'synthetic/a/ego.npz', allow_pickle=False) as saved:
        assert saved['pairwise_similarity'].shape == (6, 6)
        assert saved['scene_similarity'].shape == ()
        assert saved['scene_similarity_valid'].item()
        np.testing.assert_array_equal(saved['pairwise_similarity'], result['pairwise_similarity'][0])
