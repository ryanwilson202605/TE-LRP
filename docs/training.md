# Train AutoBot

[Home](../README.md) · [Data preparation](data-preparation.md) · [Training](training.md) · [Attribution](attribution.md)

Run all commands from the repository root after [installation](../README.md#installation).

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

For databases stored outside the repository, use absolute paths in the same overrides. For attribution, override `val_data_path` only. For the separate databases produced by the [data preparation guide](data-preparation.md):

```bash
python -m te_lrp.train 'train_data_path=[./data/converted/av2/train,./data/converted/waymo/train,./data/converted/nuscenes/train]' 'val_data_path=[./data/converted/av2/val,./data/converted/waymo/val,./data/converted/nuscenes/val]'
```

The first read writes caches. Training writes to `outputs/checkpoints/` and `outputs/logs/`. `latest.ckpt` supports resuming training; `best.ckpt` is selected by validation `brier_fde`. Validation runs without optimizer updates.

```bash
python -m te_lrp.train ckpt_path=./outputs/checkpoints/latest.ckpt
```

Use a separate `output_root` for each new training run. Training checkpoints are restored only when `ckpt_path` is set. `method.max_epochs` is the total epoch count. For a smoke test, set `max_data_num=1 method.max_epochs=1`; `max_data_num` limits the number of source scenes read per data directory.
