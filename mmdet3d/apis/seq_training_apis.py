"""
Following codes are directly copy from mmdetection-2.14.0
"""
import torch
import copy
import platform
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler
from torch.utils.data import Sampler
from collections import defaultdict

import random
import warnings


class NuScenesSceneSequentialSampler(Sampler):
    """Yields dataset indices scene-by-scene in temporal order.

    Within each scene frames are ordered by timestamp (ascending).
    Scenes themselves are shuffled each epoch so the model sees different
    scene-orderings across epochs.  This guarantees that when frame t is
    processed, frames t-1 / t-2 / t-3 of the *same* scene were processed
    in the immediately preceding steps — which is the requirement for the
    rolling Gaussian buffer to have cache hits.

    Args:
        dataset: NuScenes dataset; must expose dataset.data_infos with
                 'scene_token' and 'timestamp' per entry.
        shuffle_scenes (bool): Shuffle scene order each epoch. Default True.
        seed (int): Base seed for scene shuffling. Default 0.
    """

    def __init__(self, dataset, shuffle_scenes=True, seed=0):
        self.dataset = dataset
        self.shuffle_scenes = shuffle_scenes
        self.seed = seed
        self.epoch = 0

        # Unwrap dataset wrappers (e.g. CBGSDataset) to reach data_infos
        inner = dataset
        while not hasattr(inner, 'data_infos') and hasattr(inner, 'dataset'):
            inner = inner.dataset
        data_infos = inner.data_infos

        # Build outer_idx → inner_idx mapping (CBGSDataset remaps via sample_indices)
        if hasattr(dataset, 'sample_indices'):
            outer_to_inner = dataset.sample_indices  # list: outer_idx → inner_idx
        else:
            outer_to_inner = list(range(len(dataset)))

        # Group outer indices by scene_token, sorted by timestamp within each scene
        scene_to_indices = defaultdict(list)
        for outer_idx, inner_idx in enumerate(outer_to_inner):
            info = data_infos[inner_idx]
            scene_to_indices[info['scene_token']].append(
                (info['timestamp'], outer_idx)
            )
        # Sort each scene by timestamp and strip the timestamp
        self.scenes = []
        for token, entries in scene_to_indices.items():
            entries.sort(key=lambda x: x[0])
            self.scenes.append([idx for _, idx in entries])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        scenes = list(self.scenes)  # shallow copy
        if self.shuffle_scenes:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(scenes)
        for scene_indices in scenes:
            yield from scene_indices

    def __len__(self):
        return sum(len(s) for s in self.scenes)

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger

def build_dataloader_seq(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     logger=None,
                     weak_shuffle_cfg=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if weak_shuffle_cfg is not None and weak_shuffle_cfg['enable']:
        assert not shuffle

    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(
                dataset, samples_per_gpu, world_size, rank, seed=seed)
            logger.info('Use original DistributedGroupSampler')
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False, seed=seed, weak_shuffle_cfg=weak_shuffle_cfg)
            logger.info(f'Use DistributedSampler with weak_shuffle_cfg:{weak_shuffle_cfg}')
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if shuffle:
            sampler = GroupSampler(dataset, samples_per_gpu)
        else:
            sampler = NuScenesSceneSequentialSampler(dataset, shuffle_scenes=True, seed=0)
            logger.info('Using NuScenesSceneSequentialSampler for temporal buffer training')
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader



def train_detector_seq(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None,
                   weak_shuffle_cfg=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    shuffle = cfg.data.get('training_shuffle', False)
    logger.info(f'Training Shuffle: {shuffle}')
    logger.info(f'Weak shuffling cfg: {weak_shuffle_cfg}')

    data_loaders = [
        build_dataloader_seq(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            weak_shuffle_cfg=weak_shuffle_cfg,
            shuffle=shuffle,
            logger=logger,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())
    else:
        # Update scene sampler epoch so scene order is reshuffled each epoch
        if not shuffle and hasattr(data_loaders[0].sampler, 'set_epoch'):
            from mmcv.runner import HOOKS, Hook
            @HOOKS.register_module(name='SceneSamplerEpochHook', force=True)
            class SceneSamplerEpochHook(Hook):
                def __init__(self, sampler):
                    self.sampler = sampler
                def before_epoch(self, runner):
                    self.sampler.set_epoch(runner.epoch)
            runner.register_hook(SceneSamplerEpochHook(data_loaders[0].sampler))

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)