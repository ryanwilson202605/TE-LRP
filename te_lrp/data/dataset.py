import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf
import pickle
from collections import defaultdict
import h5py
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from te_lrp.data import preprocessing
from te_lrp.data.preprocessing import get_polyline_dir, find_true_segments, generate_mask, interpolate_polyline
from te_lrp.data.types import object_type, polyline_type

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)


class AutoBotDataset(Dataset):

    """Load ScenarioNet scenes, preprocess AutoBot inputs, and cache them in HDF5."""
    def __init__(self, config, is_validation=False):
        """Initialize data sources and cache from config (DictConfig) and is_validation (bool); returns None."""
        self.config = config
        self.data_path = config.val_data_path if is_validation else config.train_data_path
        self.file_cache = {}
        self.starting_frame = config.starting_frame
        self.load_data()

    def load_data(self):
        """Load scenes using caches isolated by source and preprocessing settings.

        Inputs: self (AutoBotDataset): configured data sources and preprocessing options.
        Returns: None; populates the sample index and reports invalid scenes."""
        self.data_loaded = {}
        # Isolate caches by source and preprocessing settings.
        keys = ("past_len", "future_len", "trajectory_sample_interval", "object_type",
                "only_train_on_ego", "line_type", "masked_attributes", "center_offset_of_map",
                "max_num_agents", "max_num_roads", "max_points_per_lane", "map_range",
                "starting_frame", "max_data_num")
        settings = {key: OmegaConf.to_container(self.config, resolve=True)[key] for key in keys}
        for source in self.data_path:
            data_path = Path(source).resolve()
            signature = json.dumps({"source": str(data_path), "settings": settings, "format": 2}, sort_keys=True)
            cache_id = hashlib.sha256(signature.encode()).hexdigest()[:20]
            cache_dir = Path(self.config.cache_path) / cache_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            index_path = cache_dir / "index.pkl"
            hdf5_path = cache_dir / "samples.h5"
            if index_path.exists():
                with index_path.open("rb") as stream:
                    groups = pickle.load(stream)
            else:
                _, summary, mapping = read_dataset_summary(str(data_path))
                groups = []
                # Write the index last so interrupted caches are rebuilt.
                with h5py.File(hdf5_path, "w") as output_file:
                    for file_name in tqdm(list(summary)[:self.config.max_data_num], desc=data_path.name):
                        scenario = read_scenario(str(data_path), mapping, file_name)
                        timestamps = np.asarray(scenario["metadata"]["ts"])
                        if not np.allclose(np.diff(timestamps), 0.1, atol=1e-4):
                            raise ValueError(f"Scene {file_name} does not use a 0.1-second sampling interval")
                        if len(timestamps) < self.starting_frame + self.config.past_len + self.config.future_len:
                            raise ValueError(f"Scene {file_name} is too short for starting_frame={self.starting_frame}, "
                                             f"past_len={self.config.past_len}, future_len={self.config.future_len}")
                        records = self.process(internal_format=self.preprocess(scenario=scenario))
                        if records is None:
                            continue
                        for record in records:
                            group_name = str(len(groups))
                            group = output_file.create_group(group_name)
                            for key, value in record.items():
                                value = np.asarray(value)
                                if value.dtype.kind == "U":
                                    value = np.asarray(value, dtype=h5py.string_dtype("utf-8"))
                                group.create_dataset(key, data=value)
                            groups.append(group_name)
                with index_path.open("wb") as stream:
                    pickle.dump(groups, stream)
            for group_name in groups:
                self.data_loaded[(cache_id, group_name)] = (hdf5_path, group_name)
        self.data_loaded_keys = list(self.data_loaded)
        if not self.data_loaded_keys:
            raise ValueError("No usable scenes; check data paths, target vehicles, and scene limits")

    def preprocess(self, scenario):

        """Convert a ScenarioNet scene into trajectory and map representations.

        Inputs: scenario (dict): tracks, map_features, and metadata.
        Returns: dict: time window, target indices, trajectories, and map points."""
        tracks = scenario['tracks']
        map_feat = scenario['map_features']

        past_length = self.config.past_len
        future_length = self.config.future_len
        total_steps = past_length + future_length
        starting_frame = self.starting_frame
        ending_frame = starting_frame + total_steps
        trajectory_sample_interval = self.config.trajectory_sample_interval
        frequency_mask = generate_mask(current_index=past_length - 1, total_length=total_steps, interval=trajectory_sample_interval)

        track_infos = {
            'object_id': [],
            'object_type': [],
            'trajs': []
        }

        for k, v in tracks.items():

            state = v['state']
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [state['position'], state['length'], state['width'], state['height'], state['heading'],
                         state['velocity'], state['valid']]
            # x, y, z, length, width, height, heading, vx, vy, valid
            all_state = np.concatenate(all_state, axis=-1)
            if all_state.shape[0] < ending_frame:
                all_state = np.pad(all_state, ((ending_frame - all_state.shape[0], 0), (0, 0)))
            all_state = all_state[starting_frame:ending_frame]

            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'

            track_infos['object_id'].append(k)
            track_infos['object_type'].append(object_type[v['type']])
            track_infos['trajs'].append(all_state)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]
        scenario['metadata']['ts'] = scenario['metadata']['ts'][starting_frame:ending_frame]

        map_infos = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'stop_sign': [],
            'crosswalk': [],
            'speed_bump': [],
        }
        polylines = []
        point_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v['type']]
            if polyline_type_ == 0:
                continue

            cur_info = {'id': k}
            cur_info['type'] = v['type']
            if polyline_type_ in [1, 2, 3]:
                polyline = v['polyline']
                if self.config.max_points_per_lane > polyline.shape[0]:
                    polyline = interpolate_polyline(polyline=polyline)
                map_infos['lane'].append(cur_info)
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
                polyline = v['polyline'] if 'polyline' in v else v['polygon']
                polyline = interpolate_polyline(polyline=polyline)
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [15, 16]:
                polyline = v['polyline']
                polyline = interpolate_polyline(polyline=polyline)
                cur_info['type'] = 7
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [17]:
                cur_info['lane_ids'] = v['lane']
                cur_info['position'] = v['position']
                map_infos['stop_sign'].append(cur_info)
                polyline = v['position'][np.newaxis]
            elif polyline_type_ in [18]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            elif polyline_type_ in [19]:
                map_infos['crosswalk'].append(cur_info)
                polyline = v['polygon']
            if polyline.shape[-1] == 2:
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            if len(polyline) == 0:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            else:
                cur_polyline_dir = get_polyline_dir(polyline=polyline)
                type_array = np.full((len(polyline), 1), polyline_type_)
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
            polylines.append(cur_polyline)
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        map_infos['all_polylines'] = (np.concatenate(polylines, axis=0).astype(np.float32)
                                      if polylines else np.zeros((0, 7), dtype=np.float32))
        ret = {'track_infos': track_infos, 'map_infos': map_infos}
        ret.update(scenario['metadata'])
        ret['timestamps_seconds'] = ret.pop('ts')
        ret['current_time_index'] = self.config.past_len - 1
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id'])

        if self.config.only_train_on_ego:
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],
                'difficulty': [0],
                'object_type': [MetaDriveType.VEHICLE]
            }
        elif ret.get('tracks_to_predict', None) is None:
            filtered_tracks = self.trajectory_filter(data=ret)
            sample_list = list(filtered_tracks.keys())
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
            }
        else:
            sample_list = list(ret['tracks_to_predict'].keys())  # + ret.get('objects_of_interest', [])
            sample_list = sorted(set(sample_list))
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
            }

        ret['tracks_to_predict'] = tracks_to_predict

        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]

        ret['track_length'] = total_steps
        return ret

    def process(self, internal_format):

        """Build model samples in the target vehicle coordinate frame.

        Inputs: internal_format (dict): a preprocessed scene.
        Returns: list[dict] or None: one sample per prediction target."""
        info = internal_format
        scene_id = info['scenario_id']

        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )
        if center_objects is None: return None

        sample_num = center_objects.shape[0]

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
         obj_trajs_future_mask, center_gt_trajs,
         center_gt_trajs_mask, center_gt_final_valid_idx,
         track_index_to_predict_new, agent_indices) = self.get_agent_data(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types
        )

        ret_dict = {
            'agent_ids': np.where(agent_indices >= 0, np.asarray(track_infos['object_id'])[agent_indices.clip(min=0)], ''),
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        map_polylines_data, map_polylines_mask, map_polylines_center = self.get_map_data(
            center_objects=center_objects, map_infos=info['map_infos'])

        ret_dict['map_polylines'] = map_polylines_data
        ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict['map_polylines_center'] = map_polylines_center

        # masking out unused attributes to Zero
        masked_attributes = self.config.masked_attributes
        if 'z_axis' in masked_attributes:
            ret_dict['obj_trajs'][..., 2] = 0
            ret_dict['map_polylines'][..., 2] = 0
        if 'size' in masked_attributes:
            ret_dict['obj_trajs'][..., 3:6] = 0
        if 'velocity' in masked_attributes:
            ret_dict['obj_trajs'][..., 25:27] = 0
        if 'acceleration' in masked_attributes:
            ret_dict['obj_trajs'][..., 27:29] = 0
        if 'heading' in masked_attributes:
            ret_dict['obj_trajs'][..., 23:25] = 0

        # change every thing to float32
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
        ret_dict['dataset_name'] = [info['dataset']] * sample_num

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {}
            for k, v in ret_dict.items():
                ret_dict_i[k] = v[i]
            ret_list.append(ret_dict_i)

        return ret_list

    def collate_fn(self, data_list):
        """Collate a list of sample dicts into a batch dict, converting numeric arrays to tensors."""
        input_dict = {}
        for key in data_list[0]:
            values = np.stack([record[key] for record in data_list])
            input_dict[key] = values if values.dtype.kind in "OUS" else torch.from_numpy(values)
        return {"batch_size": len(data_list), "input_dict": input_dict}

    def __len__(self):
        """Return the number of cached samples as an int."""
        return len(self.data_loaded_keys)

    def __getitem__(self, idx):
        """Read a cached sample by integer index; return its field dict."""
        file_path, group_name = self.data_loaded[self.data_loaded_keys[idx]]
        if file_path not in self.file_cache:
            self.file_cache[file_path] = h5py.File(file_path, "r")
        group = self.file_cache[file_path][group_name]
        record = {}
        for key, dataset in group.items():
            record[key] = dataset.asstr()[()] if h5py.check_string_dtype(dataset.dtype) else dataset[()]
        return record

    def close_files(self):
        """Close cached HDF5 handles; return None."""
        for stream in self.file_cache.values():
            stream.close()
        self.file_cache.clear()

    def get_agent_data(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types
    ):

        """Select nearby agents and construct history and future tensors.

        Inputs: center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, timestamps, obj_types (ndarray); sdc_track_index (int).
        Returns: tuple[ndarray, ...]: features, masks, labels, target indices, and agent indices."""
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.config.max_num_agents
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

        agent_indices = np.pad(topk_idxs, ((0, 0), (0, max_num_agents - topk_idxs.shape[1])), constant_values=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = (topk_idxs[..., 0, 0] == track_index_to_predict[:, None]).argmax(axis=1)

        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask.astype(bool), center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new, agent_indices)

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        """Select valid prediction targets of the configured object types.

        Inputs: track_index_to_predict, obj_trajs_full, obj_types (ndarray); current_time_index (int), scene_id (str).
        Returns: tuple[ndarray | None, ndarray | list]: target states and source indices."""
        center_objects_list = []
        track_index_to_predict_selected = []
        selected_type = self.config.object_type
        selected_type = [object_type[x] for x in selected_type]
        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
                print(f'Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}')
                continue
            if obj_types[obj_idx] not in selected_type:
                print(f'Warning: object type in data is not a selected type for scene_id={scene_id}')
                continue

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)
        if len(center_objects_list) == 0:
            print(f'Warning: no center objects at time step {current_time_index}, scene_id={scene_id}')
            return None, []
        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    def transform_trajs_to_center_coords(self, obj_trajs, center_xyz, center_heading, heading_index,
                                         rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = preprocessing.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = preprocessing.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
                angle=-center_heading
            ).reshape(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def get_map_data(self, center_objects, map_infos):

        """Select map polylines and transform them into model inputs.

        Inputs: center_objects (ndarray), map_infos (dict).
        Returns: tuple[ndarray, ndarray, ndarray]: polyline features, validity masks, and centers."""
        num_center_objects = center_objects.shape[0]

        def transform_to_center_coordinates(neighboring_polylines):
            """Transform map positions and directions into the target frame.

            Inputs: neighboring_polylines (ndarray): map features.
            Returns: ndarray: transformed positions and directions."""
            neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
            neighboring_polylines[:, :, 0:2] = preprocessing.rotate_points_along_z(
                points=neighboring_polylines[:, :, 0:2],
                angle=-center_objects[:, 6]
            )
            neighboring_polylines[:, :, 3:5] = preprocessing.rotate_points_along_z(
                points=neighboring_polylines[:, :, 3:5],
                angle=-center_objects[:, 6]
            )

            return neighboring_polylines

        polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)

        map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
        num_of_src_polylines = self.config.max_num_roads
        map_infos['polyline_transformed'] = map_polylines

        all_polylines = map_infos['polyline_transformed']
        max_points_per_lane = self.config.max_points_per_lane
        line_type = self.config.line_type
        map_range = self.config.map_range
        center_offset = self.config.center_offset_of_map
        num_agents = all_polylines.shape[0]
        polyline_list = []
        polyline_mask_list = []

        for k, v in map_infos.items():
            if k == 'all_polylines' or k not in line_type:
                continue
            if len(v) == 0:
                continue
            for polyline_dict in v:
                polyline_index = polyline_dict.get('polyline_index', None)
                polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
                polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
                polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
                in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

                segment_index_list = []
                for i in range(polyline_segment.shape[0]):
                    segment_index_list.append(find_true_segments(mask=in_range_mask[i]))
                max_segments = max([len(x) for x in segment_index_list])

                segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
                segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)

                for i in range(polyline_segment.shape[0]):
                    if in_range_mask[i].sum() == 0:
                        continue
                    segment_i = polyline_segment[i]
                    segment_index = segment_index_list[i]
                    for num, seg_index in enumerate(segment_index):
                        segment = segment_i[seg_index]
                        if segment.shape[0] > max_points_per_lane:
                            segment_list[i, num] = segment[
                                np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)]
                            segment_mask_list[i, num] = 1
                        else:
                            segment_list[i, num, :segment.shape[0]] = segment
                            segment_mask_list[i, num, :segment.shape[0]] = 1

                polyline_list.append(segment_list)
                polyline_mask_list.append(segment_mask_list)
        if not polyline_list:
            return (np.zeros((num_agents, num_of_src_polylines, max_points_per_lane, 29), dtype=np.float32),
                    np.zeros((num_agents, num_of_src_polylines, max_points_per_lane), dtype=bool),
                    np.zeros((num_agents, num_of_src_polylines, 3), dtype=np.float32))
        batch_polylines = np.concatenate(polyline_list, axis=1)
        batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

        polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
        polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
            batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
        polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
        topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

        # Ensure topk_idxs has the correct shape for indexing
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
        map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)

        # pad map_polylines and map_polylines_mask to num_of_src_polylines
        map_polylines = np.pad(map_polylines,
                               ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(
            axis=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0,
                                                  a_max=None)  # (num_center_objects, num_polylines, 3)

        xy_pos_pre = map_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

        map_types = map_polylines[:, :, :, -1]
        map_polylines = map_polylines[:, :, :, :-1]
        # one-hot encoding for map types, 14 types in total, use 20 for reserved types
        map_types = np.eye(20)[map_types.astype(int)]

        map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
        map_polylines[map_polylines_mask == 0] = 0

        return map_polylines, map_polylines_mask, map_polylines_center

    def trajectory_filter(self, data):

        """Select prediction candidates using history validity and motion state.

        Inputs: data (dict): trajectories and object summaries.
        Returns: dict: candidate IDs mapped to object indices and types."""
        trajs = data['track_infos']['trajs']
        current_idx = data['current_time_index']
        obj_summary = data['object_summary']

        tracks_to_preidct = {}
        for idx,(k,v) in enumerate(obj_summary.items()):
            type = v['type']
            validity = trajs[idx, :, -1]
            if type not in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']: continue
            valid_ratio = v['valid_length']/v['track_length']
            if valid_ratio < 0.5: continue
            moving_distance = v['moving_distance']
            if moving_distance < 2.0 and type=='VEHICLE': continue
            is_valid_at_m = validity[current_idx]>0
            if not is_valid_at_m: continue

            tracks_to_preidct[k] = {'track_index': idx, 'track_id': k, 'difficulty': 0, 'object_type': type}

        return tracks_to_preidct
