# Data preparation

[Home](../README.md) · [Data preparation](data-preparation.md) · [Training](training.md) · [Attribution](attribution.md)

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
