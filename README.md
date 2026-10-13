# TE-LRP

**Trajectory Endpoint Layer-wise Relevance Propagation for AutoBot**

TE-LRP explains each prediction mode's x and y endpoint coordinates separately. Signed input relevance is aggregated into two-dimensional agent and map attribution vectors, magnitudes, and normalized distributions. Cross-mode attribution similarity is computed from these distributions using the complement of the base-2 Jensen-Shannon distance.

This repository presents the method implementation of TE-LRP for **AutoBot**: predicted-endpoint relevance propagation, the input-independent query (IIQ) rule, entity-level attribution aggregation, and cross-mode attribution similarity. Data loading, prediction, and training utilities support running the method on **AV2, Waymo Open Motion, and nuScenes** in ScenarioNet PKL format.

**Official data download → ScenarioNet conversion → PKL preprocessing and caching → AutoBot prediction → TE-LRP attribution and similarity**

Use the included checkpoint or train your own AutoBot model. All commands below run from the repository root.

## Installation

Run these commands from the repository root:

```bash
conda env create -f environment.yml
```

```bash
conda activate te-lrp
```

The tested runtime uses Python 3.9, PyTorch 2.8.0, and NumPy 2.0.2. Git is included to install ScenarioNet from its pinned source commit. All direct runtime dependencies are listed in [environment.yml](environment.yml). After activating the environment, run the entry points from the repository root; installing this repository as a package is unnecessary. Choose a PyTorch CUDA wheel compatible with your machine, or add `device=cpu` to run on CPU.

## Data preparation

Use a separate conversion environment. Follow the [ScenarioNet installation guide](https://scenarionet.readthedocs.io/en/latest/install.html) and use the pinned commit shown below. The official Waymo example uses TensorFlow 2.11 and Protobuf 3.20, whose NumPy requirements differ from this runtime. AV2 requires the `av2` devkit; nuScenes requires `nuscenes-devkit`, maps, and the CAN bus expansion.

### Conversion environment

Install the official converters and dataset parsers in a separate environment to avoid conflicting TensorFlow and runtime NumPy requirements:

```bash
conda create -n scenarionet-convert python=3.9 git -y
```

```bash
conda activate scenarionet-convert
```

```bash
python -m pip install "numpy==1.23.5" "tensorflow==2.11.0" "protobuf==3.20.3" "setuptools<81" "metadrive-simulator==0.4.3" av2 nuscenes-devkit "git+https://github.com/metadriverse/scenarionet.git@32910a3a1b51ef9e8d4a79a0a27b8066bd1214c9"
```

These installation commands were assembled from upstream documentation and have not been tested in a fresh environment. After conversion, switch back with `conda activate te-lrp`.

### Convert official raw data to PKL

This stage uses ScenarioNet commands. Download and extract the official data and required maps, and replace `/absolute/path/...` with actual absolute paths. Outputs use the suggested `./data/converted/` layout. The runtime configuration currently points to existing merged databases; override its source paths to read these newly converted directories. Run conversion commands from a working directory without subdirectories named `scenarionet` or `metadrive`.

| Dataset | Official sources and conversion instructions |
|---|---|
| AV2 Motion Forecasting | [Download guide](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data), [ScenarioNet AV2](https://scenarionet.readthedocs.io/en/latest/argoverse2.html) |
| Waymo Open Motion | [ScenarioNet Waymo example and download link](https://scenarionet.readthedocs.io/en/latest/example.html#waymo-example); use scenario TFRecords, not perception data or tf.Example files |
| nuScenes | [Official download](https://www.nuscenes.org/nuscenes), [ScenarioNet nuScenes](https://scenarionet.readthedocs.io/en/latest/nuscenes.html); prepare trainval metadata, maps, and CAN bus while retaining the official directory structure |

#### AV2

```bash
python -m scenarionet.convert_argoverse2 --raw_data_path /absolute/path/av2/train -d ./data/converted/av2/train
```

```bash
python -m scenarionet.convert_argoverse2 --raw_data_path /absolute/path/av2/val -d ./data/converted/av2/val
```

These commands convert all Parquet scenes found recursively under each input directory. In the pinned ScenarioNet version, the AV2 file-discovery function does not apply `--num_files` or `--start_file_index`; use a directory containing only the desired scenes to limit conversion.

#### Waymo

```bash
python -m scenarionet.convert_waymo --raw_data_path /absolute/path/waymo/train -d ./data/converted/waymo/train
```

```bash
python -m scenarionet.convert_waymo --raw_data_path /absolute/path/waymo/val -d ./data/converted/waymo/val
```

The converter reads TFRecord files directly inside the input directory, without recursive discovery. Use a directory containing only scenario TFRecords; add `--num_files 1` to process one file.

#### nuScenes

```bash
python -m scenarionet.convert_nuscenes --dataroot /absolute/path/nuscenes --split train -d ./data/converted/nuscenes/train
```

```bash
python -m scenarionet.convert_nuscenes --dataroot /absolute/path/nuscenes --split val -d ./data/converted/nuscenes/val
```

`train` and `val` use the converter's prediction splits, with a default 2-second history and 6-second future interpolated at 0.1-second intervals. `v1.0-trainval` identifies the full log collection, rather than this prediction split. Use `mini_train` or `mini_val` for mini checks.

Retain `dataset_summary.pkl`, `dataset_mapping.pkl`, and every referenced scene PKL. Configure database root paths containing these two indexes. Directories such as `*_0` and `*_1` are worker storage shards, not official train/val splits.

### Merge and split PKL databases

The loader supports multiple data directories directly. The current defaults point to existing merged training and validation databases. To combine separate dataset directories at load time, pass all training sources in `train_data_path` and all validation sources in `val_data_path`. Physical merging is unnecessary; an existing merged database can also be supplied as a single path.

The following [ScenarioNet database operations](https://scenarionet.readthedocs.io/en/latest/operations.html) are optional. Merge training and validation sources separately:

```bash
python -m scenarionet.merge --from ./data/converted/av2/train ./data/converted/waymo/train ./data/converted/nuscenes/train --to ./data/merged_train
```

```bash
python -m scenarionet.merge --from ./data/converted/av2/val ./data/converted/waymo/val ./data/converted/nuscenes/val --to ./data/merged_val
```

`merge` creates combined indexes referencing source PKLs through relative paths. Keep the source directories and do not move only the merged directory.

For a custom split of an existing database, the following commands use the official example's 40/21 scene counts and require at least 61 source scenes. This custom split does not replace official train/val splits:

```bash
python -m scenarionet.split --from ./data/converted/example --to ./data/custom_train --num_scenarios 40 --start_index 0
```

```bash
python -m scenarionet.split --from ./data/converted/example --to ./data/custom_val --num_scenarios 21 --start_index 40
```

```bash
python -m scenarionet.check_overlap --d_1 ./data/custom_train --d_2 ./data/custom_val
```

```bash
python -m scenarionet.num -d ./data/custom_train
```

`split` also writes indexes referencing source scenes. These commands select disjoint ranges in index order without `--random`.

## Train AutoBot

Current defaults use `shared_root=/mnt/disk1/shared_data`, with `${shared_root}/data/merged_train` for training and `${shared_root}/data/merged_val` for validation and attribution. These paths refer to existing local databases. Override `shared_root` if your merged databases follow the same layout elsewhere.

`te_lrp/configs/config.yaml` specifies data sources, caches, devices, and outputs. `te_lrp/configs/method/autobot.yaml` specifies model and training parameters. Defaults use 21 history frames, 60 future frames, six modes, 15 agents, and 256 map entities. Positions are expressed in the target vehicle's local frame. Prediction targets the ScenarioNet SDC by default; set `only_train_on_ego=false` to use the scene's prediction-target list, or the loader's motion/validity filter when that list is absent. `starting_frame=0` selects the first 81 frames of each converted scene; change this value to shift the window.

Map preprocessing interpolates selected map features, transforms points into the target frame, and selects contiguous runs of points inside the configured region. It does not compute geometric intersections at the region boundary. Segments longer than `max_points_per_lane` are sampled to that point count; shorter segments are padded and masked. `manually_split_lane`, `point_sampled_interval`, `num_points_each_polyline`, and `vector_break_dist_thresh` are not supported in this release.

HDF5 caches are created on first use and reused thereafter. Caches are isolated by source path and preprocessing settings. Use a new `cache_path` after changing PKL contents. Preprocessing leaves source files unchanged. Keep both database indexes and all referenced scene files: the upstream reader can infer missing indexes for flat directories, but sharded databases need their mappings. Missing referenced files, timestamps not spaced at 0.1 seconds, and scenes shorter than the required window raise errors when building a cache. Training uses a single device.

```bash
python -m te_lrp.train device=cuda
```

To use merged databases under the repository directory, override both source lists:

```bash
python -m te_lrp.train 'train_data_path=[./data/merged_train]' 'val_data_path=[./data/merged_val]'
```

For databases stored outside the repository, use absolute paths in the same overrides. For attribution, override `val_data_path` only. For the separate databases produced by the data preparation section above:

```bash
python -m te_lrp.train 'train_data_path=[./data/converted/av2/train,./data/converted/waymo/train,./data/converted/nuscenes/train]' 'val_data_path=[./data/converted/av2/val,./data/converted/waymo/val,./data/converted/nuscenes/val]'
```

The first read writes caches. Training writes to `outputs/checkpoints/` and `outputs/logs/`. `latest.ckpt` supports resuming training; `best.ckpt` is selected by validation `brier_fde`. Validation runs without optimizer updates.

```bash
python -m te_lrp.train ckpt_path=./outputs/checkpoints/latest.ckpt
```

Use a separate `output_root` for each new training run. Training checkpoints are restored only when `ckpt_path` is set. `method.max_epochs` is the total epoch count. For a smoke test, set `max_data_num=1 method.max_epochs=1`; `max_data_num` limits the number of source scenes read per data directory.

## Run TE-LRP

```bash
python -m te_lrp.attribution max_data_num=1 attribution.batch_limit=1
```

This uses the configured merged validation database and included AutoBot checkpoint, creates preprocessing caches, and saves predictions, attributions, and attribution similarities. `max_data_num=1` limits preprocessing to one source scene per data directory; `attribution.batch_limit=1` limits attribution to one batch, not preprocessing. Remove both overrides to process the full dataset.

Set `eval_ckpt_path=./outputs/checkpoints/best.ckpt` to use your own trained weights. Use `device=cpu`, `attribution.batch_size`, `attribution.batch_start`, and `attribution.attribution_dir` to configure execution. Training and attribution must use the same `method.*` architecture settings, `past_len`, and `future_len`. The included [AutoBot checkpoint](checkpoints/autobot.ckpt) uses the default architecture and 21/60-frame history/future lengths. It is provided for prediction and attribution; resume training with checkpoints saved by this repository through `ckpt_path`.

TE-LRP explains the final predicted x and y coordinates separately, seeding relevance with each signed coordinate value. The attribution computation does not use ground truth, FDE, or hit/miss labels; the shared data loader still requires the complete configured history/future window. All modes share one forward pass. IIQ applies only to calls with input-independent queries; queries updated through scene interactions use ordinary attention propagation. Epsilon is `1e-6`.

Import `enable_te_lrp` from `te_lrp.propagation` and `attribute` from `te_lrp.propagation.endpoint`. Load weights, call `enable_te_lrp(model=model)`, then `attribute(model=model, input_dict=input_dict)`. The enabled instance is for attribution only. Other model instances and global PyTorch classes are unaffected. `model.predict(batch=batch)` returns predictions without computing training loss.

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

## Repository structure

```text
te_lrp/
├── propagation/     # TE-LRP endpoint attribution and propagation rules
│   ├── endpoint.py  # Endpoint initialization and entity aggregation
│   ├── autobot.py   # Enable propagation rules on AutoBot
│   ├── layers.py    # Layer implementations for relevance propagation
│   └── rules.py     # Basic propagation rules
├── models/          # AutoBot
├── data/            # PKL loading, preprocessing, and caching
├── configs/         # Hydra configurations
├── training/        # Training loop, validation metrics, and logging
├── similarity.py    # Pairwise attribution similarity and scene-level mean
├── train.py         # Training entry point
└── attribution.py   # Prediction and attribution entry point
checkpoints/         # Included AutoBot checkpoint
licenses/            # Original upstream license texts
tests/               # Implementation tests
```

## Testing

Install the optional test dependencies after creating the runtime environment:

```bash
python -m pip install -r requirements-dev.txt
```

```bash
python -m pytest -q tests
```

Tests cover propagation rules, IIQ, prediction equivalence, ground-truth independence, entity aggregation, attribution similarity, data caching, and checkpoint restoration.

## Acknowledgments and licensing

The original TE-LRP code and the authors-trained checkpoint at `checkpoints/autobot.ckpt` are released under the **GNU Affero General Public License, version 3 or any later version (AGPL-3.0-or-later)**; see [LICENSE](LICENSE). Third-party portions retain their original copyright and license notices listed below. Dataset downloads remain subject to their providers' terms.

This repository modifies the upstream AutoBot integration and preprocessing for standalone training, caching, endpoint attribution, and similarity computation. These modifications are distributed as part of TE-LRP (2026).

| Source | Use in this repository | Upstream license |
|---|---|---|
| [UniTraj](https://github.com/vita-epfl/UniTraj) | AutoBot integration, model infrastructure, and preprocessing (`te_lrp/models/`, `te_lrp/data/`) | [Upstream notice](licenses/UniTraj.txt), [AGPL v3 text](LICENSE); the notice specifies AGPLv3-or-later |
| [AutoBots](https://github.com/roggirg/AutoBots) | AutoBot model source | [BSD 3-Clause](licenses/AutoBots.txt) |
| [ScenarioNet](https://github.com/metadriverse/scenarionet), [MetaDrive](https://github.com/metadriverse/metadrive) | External data-format and loading dependencies | See the licenses distributed with each dependency |

License texts and copyright notices are retained unchanged. [LICENSE](LICENSE) contains the standard GNU AGPL v3 text; the upstream notices are preserved in [licenses/](licenses/).

UniTraj-derived portions retain the full [upstream notice](licenses/UniTraj.txt), including its additional use-related wording. The license granted for original TE-LRP contributions does not replace or remove those upstream notices.
