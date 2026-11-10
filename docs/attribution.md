# Run TE-LRP

[Home](../README.md) · [Data preparation](data-preparation.md) · [Training](training.md) · [Attribution](attribution.md)

Run all commands from the repository root after [installation](../README.md#installation).

```bash
python -m te_lrp.attribution max_data_num=1 attribution.batch_limit=1
```

This uses the configured merged validation database and included AutoBot checkpoint, creates preprocessing caches, and saves predictions, attributions, and attribution similarities. `max_data_num=1` limits preprocessing to one source scene per data directory; `attribution.batch_limit=1` limits attribution to one batch, not preprocessing. Remove both overrides to process the full dataset.

Set `eval_ckpt_path=./outputs/checkpoints/best.ckpt` to use your own trained weights. Use `device=cpu`, `attribution.batch_size`, `attribution.batch_start`, and `attribution.attribution_dir` to configure execution. Training and attribution must use the same `method.*` architecture settings, `past_len`, and `future_len`. The included [AutoBot checkpoint](../checkpoints/autobot.ckpt) uses the default architecture and 21/60-frame history/future lengths. It is provided for prediction and attribution; resume training with checkpoints saved by this repository through `ckpt_path`.

## Method and Python API

TE-LRP explains the final predicted x and y coordinates separately, seeding relevance with each signed coordinate value. The attribution computation does not use ground truth, FDE, or hit/miss labels; the shared data loader still requires the complete configured history/future window. All modes share one forward pass. IIQ applies only to calls with input-independent queries; queries updated through scene interactions use ordinary attention propagation. Epsilon is `1e-6`.

Import `enable_te_lrp` from `te_lrp.propagation` and `attribute` from `te_lrp.propagation.endpoint`. Load weights, call `enable_te_lrp(model=model)`, then `attribute(model=model, input_dict=input_dict)`. The enabled instance is for attribution only. Other model instances and global PyTorch classes are unaffected. `model.predict(batch=batch)` returns predictions without computing training loss.

## Output fields

Outputs are saved to `outputs/attributions/<dataset>/<scenario>/<target>.npz`. Dataset, scene, and target IDs are URL-encoded for use in paths. Rerunning with the same output path overwrites the corresponding result. Each run saves its resolved configuration. `eval_ckpt_path` defaults to the included checkpoint; change it explicitly to use newly trained weights.

| Field | Per-target shape or meaning |
|---|---|
| `predicted_trajectory` | `[K,T_future,5]`; the first two channels are xy, followed by three AutoBot distribution parameters |
| `predicted_probability` | `[K]` |
| `agent_relevance` | `[K,2,N,T_past,F_agent]`; axis 1 identifies the explained x/y output coordinate |
| `map_relevance` | `[K,2,S,P,F_map]` |
| `entity_vectors` | `[K,N+S,2]`; signed relevance sums, with agents followed by map entities |
| `entity_magnitude` | `[K,N+S]`; L2 norms of entity vectors |
| `entity_distribution` | `[K,N+S]`; joint normalization over agents and map entities |
| `entity_valid` / `distribution_valid` | `[N+S]` / `[K]`; entity and distribution validity flags |
| `pairwise_similarity` / `pairwise_similarity_valid` | `[K,K]`; cross-mode similarities and validity flags |
| `scene_similarity` / `scene_similarity_valid` | Scalars; mean over all distinct mode pairs and validity flag |
| Inputs and identifiers | Trajectories, maps, masks, `agent_ids`, `map_entity_index`, target index, and target world-frame state |

`K`, `T_past`, `T_future`, `N`, `S`, and `P` follow `num_modes`, `past_len`, `future_len`, `max_num_agents`, `max_num_roads`, and `max_points_per_lane`, respectively. Shapes include padded entities.

Map entities are identified by preprocessed input-polyline indices; `map_entity_index` is not an original map feature ID. Padded inputs receive zero relevance. Zero total magnitude produces a NaN distribution with `distribution_valid=false`. Read results with `numpy.load(path, allow_pickle=False)`.

## Attribution similarity

The attribution command also computes pairwise similarity as one minus the square root of the base-2 Jensen-Shannon divergence between entity distributions. Identical distributions have similarity 1; disjoint supports have similarity 0. Agents and map entities remain jointly normalized. Mode probabilities and ground truth are not used.

The scene score averages all distinct mode pairs (15 pairs for six modes), excluding the diagonal. It is computed separately for each prediction target and input window, not across targets or scenes. Pairs involving an undefined distribution are NaN with a false validity flag. If any mode is invalid, or fewer than two modes exist, the scene score is NaN; invalid pairs are not silently dropped.

The Python function accepts normalized distributions shaped `[B,K,E]` and Boolean validity flags shaped `[B,K]`, where `E` contains agents followed by map entities in the same order for every mode. It returns pairwise fields shaped `[B,K,K]` and scene fields shaped `[B]`. Valid diagonal entries equal 1 and are excluded from the scene mean. No additional runtime dependencies are required.

Existing NPZ distributions can be compared without rerunning the model. The following example prints the scores and does not modify the input file:

```python
import numpy as np
import torch
from te_lrp.similarity import attribution_similarity

# Replace this path with an existing attribution output.
with np.load("outputs/attributions/dataset/scenario/target.npz", allow_pickle=False) as result:
    scores = attribution_similarity(
        entity_distribution=torch.from_numpy(result["entity_distribution"])[None],
        distribution_valid=torch.from_numpy(result["distribution_valid"])[None],
    )
print(scores["pairwise_similarity"][0])
print(scores["scene_similarity"][0].item())
```
