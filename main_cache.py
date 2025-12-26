import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import yaml
from easydict import EasyDict as edict

import torch
from torch import nn
import torch.backends.cudnn as cudnn

import utils.misc as misc
from utils.utils import ensure_path_exists

from model import models_ae
from datasets import build_dataset

from engine_generation import cache_latents


def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--config', default='configs/ae_cfg_base.yaml', help='config file path')

    return parser

def main(args):
    misc.init_distributed_mode(args.train)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.system.device)

    # fix the seed for reproducibility
    seed = args.system.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    dataset_train = build_dataset.get_dataset(args.dataset, 'train')
    dataset_train.set_load_query(True)
    dataset_train.set_load_radar(False)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.dataset.batch_size,
        num_workers=args.dataset.num_workers,
        pin_memory=args.dataset.pin_mem,
        drop_last=False,
        prefetch_factor=2,
    )

    vae:nn.Module = models_ae.__dict__[args.lidar_ae.name](N=args.dataset.lidar.num_samples)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    vae.load_state_dict(torch.load(args.lidar_ae.ckpt, map_location='cpu')[
                        'model'], strict=True)
    vae.to(device)
    print('Loaded VAE from %s' % args.lidar_ae.ckpt)

    # constuct the cache path
    cache_base_path = Path(args.lidar_ae.cache_path)
    vae_name = args.lidar_ae.name
    cache_name = args.lidar_ae.cache_name
    cache_path = cache_base_path / vae_name / cache_name
    ensure_path_exists(cache_path)
    print(f"Cache path: {cache_path}")

    print(f"Start caching Lidar VAE latents")
    start_time = time.time()
    cache_latents(
        vae,
        data_loader_train,
        device,
        cache_base_path=cache_path
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))
    print('Caching path: {}'.format(cache_path))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    configs = yaml.safe_load(open(args.config))
    configs = edict(configs)
    configs.system.output_dir = os.path.join(configs.system.output_dir, configs.system.expname)
    configs.system.log_dir = os.path.join(configs.system.log_dir, configs.system.expname)
    if configs.system.output_dir:
        # Path(configs.system.output_dir).mkdir(parents=True, exist_ok=True)
        ensure_path_exists(Path(configs.system.output_dir))
        with open(os.path.join(configs.system.output_dir, Path(args.config).name), 'w') as f:
            yaml.dump(configs, f)
    main(configs)