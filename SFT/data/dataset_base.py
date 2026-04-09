# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import os
import random
import json
import time
import traceback

import numpy as np
import torch
from .debug_trace import import_with_log, agent_log

_data_utils = import_with_log(
    "data.data_utils",
    "H1",
    "data/dataset_base.py:imports",
    "before import data.data_utils",
)
get_flattened_position_ids_interpolate = _data_utils.get_flattened_position_ids_interpolate
get_flattened_position_ids_extrapolate = _data_utils.get_flattened_position_ids_extrapolate
len2weight = _data_utils.len2weight
patchify = _data_utils.patchify
prepare_attention_mask_per_sample = _data_utils.prepare_attention_mask_per_sample

_dataset_info = import_with_log(
    "data.dataset_info",
    "H1",
    "data/dataset_base.py:imports",
    "before import data.dataset_info",
)
DATASET_INFO = _dataset_info.DATASET_INFO
get_dataset_class = _dataset_info.get_dataset_class

_DEBUG_LOG_PATH = os.environ.get("SFT_DEBUG_LOG", "")
_PINMEM_LOG_MAX = 5
_pinmem_log_count = 0


def _pinmem_debug_log(hypothesis_id: str, location: str, message: str, data: dict):
    global _pinmem_log_count
    if not _DEBUG_LOG_PATH or _pinmem_log_count >= _PINMEM_LOG_MAX:
        return
    try:
        rank = int(os.environ.get("RANK", "-1"))
    except ValueError:
        rank = -1
    if rank != 0:
        return
    payload = {
        "id": f"log_{int(time.time() * 1000)}_{os.getpid()}_{_pinmem_log_count}",
        "timestamp": int(time.time() * 1000),
        "runId": "train-pinmem-debug",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _pinmem_log_count += 1
    except Exception:
        pass

_transforms = import_with_log(
    "data.transforms",
    "H1",
    "data/dataset_base.py:imports",
    "before import data.transforms/video_utils",
)
ImageTransform = _transforms.ImageTransform
_video_utils = import_with_log(
    "data.video_utils",
    "H1",
    "data/dataset_base.py:imports",
    "before import data.transforms/video_utils",
)
FrameSampler = _video_utils.FrameSampler


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        self.debug_sample = os.environ.get("UMM_SFT_DEBUG_SAMPLE", "0").lower() in {
            "1", "true", "yes", "y", "on"
        }
        self.debug_mask = os.environ.get("UMM_SFT_DEBUG_MASK", "0").lower() in {
            "1", "true", "yes", "y", "on"
        }
        self._debug_batch_printed = False
        self._debug_mask_printed = False
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, grouped_weights = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.grouped_weights = grouped_weights
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        grouped_weights = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            grouped_weights.append(dataset_args.pop('weight', 0.0))

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            if 'dataset_names' not in dataset_args:
                raise KeyError(
                    f"`dataset_names` is required for grouped dataset `{grouped_dataset_name}`."
                )
            dataset_names = dataset_args.pop('dataset_names')
            # Support automatic inclusion of all discovered datasets, e.g.
            # dataset_names: "__all__" or dataset_names: ["__all__"].
            if dataset_names == "__all__":
                dataset_names = sorted(DATASET_INFO[grouped_dataset_name].keys())
            elif isinstance(dataset_names, list) and "__all__" in dataset_names:
                dataset_names = sorted(DATASET_INFO[grouped_dataset_name].keys())
            dataset_args['data_dir_list'] = []
            dataset_args['reference_list'] = []
            dataset_args['generation_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                dataset_args['data_dir_list'].append(meta_info['data_dir'])
                dataset_args['reference_list'].append(meta_info['reference_dir'])
                dataset_args['generation_list'].append(meta_info['generation_dir'])

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])
                
                if 'json_path' in meta_info.keys():
                    # json with jpeg
                    if 'json_path_list' not in dataset_args.keys():
                        dataset_args['json_path_list'] = [meta_info['json_path']]
                    else:
                        dataset_args['json_path_list'].append(meta_info['json_path'])

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset_cls = get_dataset_class(grouped_dataset_name)
            dataset = dataset_cls(
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, grouped_weights

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(), 
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        return data

    def __iter__(self):
        total_weights = sum(self.grouped_weights)
        assert total_weights > 0.0
        group_cumprobs = [sum(self.grouped_weights[:i + 1]) / total_weights 
                          for i in range(len(self.grouped_weights))]
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        buffer = []
        while True:
            # Ensure at least one sample from each group
            if sequence_status['curr'] == 0:
                for group_index, group_iter in enumerate(self.dataset_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            # if a sample is too long, skip it
                            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                            if num_tokens < self.max_num_tokens_per_sample:
                                sequence_status_new = self.pack_sequence(sample, sequence_status)
                                if sequence_status_new is None:
                                    print(
                                        "skip a sample with final_gen but empty final_context",
                                        flush=True,
                                    )
                                    continue
                                sequence_status = sequence_status_new
                                batch_data_indexes.append(sample['data_indexes'])
                                break
                            else:
                                print(f"skip a sample with length {num_tokens}")
                                continue

            if sequence_status['curr'] < self.prefer_buffer_before and len(buffer) > 0:
                sample = buffer.pop(0)
                sample_from_buffer = True
            else:
                # sample normally across all groups
                n = random.random()
                group_index = 0
                for i, cumprob in enumerate(group_cumprobs):
                    if n < cumprob:
                        group_index = i
                        break
                sample = next(self.dataset_iters[group_index])
                sample_from_buffer = False

            # if a sample is too long, skip it
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            if num_tokens > self.max_num_tokens_per_sample:
                print(f"skip a sample with length {num_tokens}")
                continue

            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                if len(buffer) < self.max_buffer_size and not sample_from_buffer:
                    buffer.append(sample)
                else:
                    print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                    data = self.to_tensor(sequence_status)
                    data['batch_data_indexes'] = batch_data_indexes
                    if self.debug_sample and not self._debug_batch_printed:
                        mse_count = int(data['mse_loss_indexes'].numel()) if 'mse_loss_indexes' in data else 0
                        triplet_flags = [
                            bool(item.get("debug_final_triplet_ok", False))
                            for item in batch_data_indexes
                            if isinstance(item, dict)
                        ]
                        print(
                            f"[umm_sft_debug_batch] has_mse={mse_count > 0} mse_count={mse_count} "
                            f"triplet_ok={sum(triplet_flags)}/{len(triplet_flags)}",
                            flush=True,
                        )
                        self._debug_batch_printed = True
                    yield data
                    sequence_status = self.set_sequence_status()
                    batch_data_indexes = []
                continue

            sequence_status_new = self.pack_sequence(sample, sequence_status)
            if sequence_status_new is None:
                print(
                    "skip a sample with final_gen but empty final_context",
                    flush=True,
                )
                continue
            sequence_status = sequence_status_new
            batch_data_indexes.append(sample['data_indexes'])

            if sequence_status['curr'] >= self.expected_num_tokens:
                data = self.to_tensor(sequence_status)
                data['batch_data_indexes'] = batch_data_indexes
                if self.debug_sample and not self._debug_batch_printed:
                    mse_count = int(data['mse_loss_indexes'].numel()) if 'mse_loss_indexes' in data else 0
                    triplet_flags = [
                        bool(item.get("debug_final_triplet_ok", False))
                        for item in batch_data_indexes
                        if isinstance(item, dict)
                    ]
                    print(
                        f"[umm_sft_debug_batch] has_mse={mse_count > 0} mse_count={mse_count} "
                        f"triplet_ok={sum(triplet_flags)}/{len(triplet_flags)}",
                        flush=True,
                    )
                    self._debug_batch_printed = True
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        sample_start = curr
        curr_rope_id = 0
        sample_lens = 0
        final_context_ranges = []
        final_gen_ranges = []

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0
            item_start = curr

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

            item_end = curr
            if item_end > item_start:
                local_start = item_start - sample_start
                local_end = item_end - sample_start
                if item.get('final_context', 0) == 1:
                    final_context_ranges.append((local_start, local_end))
                if item.get('final_gen', 0) == 1:
                    final_gen_ranges.append((local_start, local_end))

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sample_attention_mask = prepare_attention_mask_per_sample(split_lens, attn_modes)
            if len(final_gen_ranges) > 0 and len(final_context_ranges) == 0:
                return None
            # Restrict final generation queries to only see:
            # final context segments + final gen segment itself.
            if len(final_gen_ranges) > 0 and len(final_context_ranges) > 0:
                allowed = torch.zeros(sample_lens, dtype=torch.bool, device=sample_attention_mask.device)
                for s, e in final_context_ranges:
                    allowed[s:e] = True
                for s, e in final_gen_ranges:
                    allowed[s:e] = True
                blocked_cols = ~allowed
                for s, e in final_gen_ranges:
                    sample_attention_mask[s:e, blocked_cols] = float("-inf")
            if self.debug_mask and (self.local_rank == 0) and (not self._debug_mask_printed):
                print(
                    "[umm_sft_debug_mask] "
                    f"sample_lens={sample_lens} split_lens={split_lens} "
                    f"attn_modes={attn_modes} "
                    f"final_context_ranges={final_context_ranges} "
                    f"final_gen_ranges={final_gen_ranges}",
                    flush=True,
                )
                self._debug_mask_printed = True
            sequence_status['nested_attention_masks'].append(sample_attention_mask)
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        # region agent log
        _pinmem_debug_log(
            "HP1",
            "data/dataset_base.py:SimpleCustomBatch.pin_memory",
            "pin_memory_enter",
            {
                "use_flex": bool(self.use_flex),
                "has_nested_attention_masks": hasattr(self, "nested_attention_masks"),
                "nested_attention_masks_len": (
                    len(self.nested_attention_masks)
                    if hasattr(self, "nested_attention_masks")
                    else -1
                ),
                "has_padded_images": hasattr(self, "padded_images"),
                "has_packed_vit_tokens": hasattr(self, "packed_vit_tokens"),
                "has_packed_timesteps": hasattr(self, "packed_timesteps"),
                "has_packed_label_ids": hasattr(self, "packed_label_ids"),
            },
        )
        # endregion
        phase = "pin_text"
        try:
            self.packed_text_ids = self.packed_text_ids.pin_memory()
            self.packed_text_indexes = self.packed_text_indexes.pin_memory()
            self.packed_position_ids = self.packed_position_ids.pin_memory()

            if not self.use_flex:
                phase = "pin_nested_attention_masks"
                first_shape = None
                if len(self.nested_attention_masks) > 0:
                    first_shape = list(self.nested_attention_masks[0].shape)
                # region agent log
                _pinmem_debug_log(
                    "HP2",
                    "data/dataset_base.py:SimpleCustomBatch.pin_memory",
                    "before_pin_nested_attention_masks",
                    {
                        "nested_attention_masks_len": len(self.nested_attention_masks),
                        "first_mask_shape": first_shape,
                    },
                )
                # endregion
                self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

            if hasattr(self, 'padded_images'):
                phase = "pin_vae_inputs"
                self.padded_images = self.padded_images.pin_memory()
                self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
                self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

            if hasattr(self, 'packed_timesteps'):
                phase = "pin_timesteps"
                self.packed_timesteps = self.packed_timesteps.pin_memory()
                self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

            if hasattr(self, 'packed_vit_tokens'):
                phase = "pin_vit_inputs"
                self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
                self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
                self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
                self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

            if hasattr(self, 'packed_label_ids'):
                phase = "pin_label_inputs"
                self.packed_label_ids = self.packed_label_ids.pin_memory()
                self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
                self.ce_loss_weights = self.ce_loss_weights.pin_memory()
        except Exception as e:
            # region agent log
            _pinmem_debug_log(
                "HP4",
                "data/dataset_base.py:SimpleCustomBatch.pin_memory",
                "pin_memory_exception",
                {
                    "phase": phase,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            # endregion
            raise

        # region agent log
        _pinmem_debug_log(
            "HP3",
            "data/dataset_base.py:SimpleCustomBatch.pin_memory",
            "pin_memory_exit",
            {"phase": phase},
        )
        # endregion
        return self

    def cuda(self, device):
        try:
            global_rank = int(os.environ.get("RANK", "-1"))
        except ValueError:
            global_rank = -1
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        except ValueError:
            local_rank = -1
        base_ctx = {
            "rank": global_rank,
            "local_rank": local_rank,
            "device_arg": int(device),
        }
        # region agent log
        agent_log(
            "D7",
            "data/dataset_base.py:SimpleCustomBatch.cuda",
            "cuda_enter",
            {
                **base_ctx,
                "current_cuda_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else -1,
                "use_flex": bool(self.use_flex),
                "has_padded_images": hasattr(self, "padded_images"),
                "has_packed_vit_tokens": hasattr(self, "packed_vit_tokens"),
                "has_packed_timesteps": hasattr(self, "packed_timesteps"),
                "has_packed_label_ids": hasattr(self, "packed_label_ids"),
            },
        )
        # endregion
        # region agent log
        agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_text_begin", base_ctx)
        # endregion
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)
        # region agent log
        agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_text_end", base_ctx)
        # endregion

        if not self.use_flex:
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_attention_begin", {**base_ctx, "nested_attention_masks": len(self.nested_attention_masks)})
            # endregion
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_attention_end", {**base_ctx, "nested_attention_masks": len(self.nested_attention_masks)})
            # endregion

        if hasattr(self, 'padded_images'):
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_vae_begin", base_ctx)
            # endregion
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_vae_end", base_ctx)
            # endregion

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_vit_begin", base_ctx)
            # endregion
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_vit_end", base_ctx)
            # endregion

        if hasattr(self, 'packed_label_ids'):
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_label_begin", base_ctx)
            # endregion
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)
            # region agent log
            agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "move_label_end", base_ctx)
            # endregion
        # region agent log
        agent_log("D7", "data/dataset_base.py:SimpleCustomBatch.cuda", "cuda_exit", base_ctx)
        # endregion

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
