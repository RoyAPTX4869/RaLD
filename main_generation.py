import argparse
import datetime
import json
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # for debugging
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # for debugging
import time
from pathlib import Path
import yaml
from easydict import EasyDict as edict

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.utils import ensure_path_exists

from model import models_ae, models_radar_encoder, models_radar_generation
from datasets import build_dataset

from engine_generation import train_one_epoch, evaluate




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

    dataset_train = build_dataset.get_dataset(args.dataset, 'train')
    if args.eval.get('use_test_set', False):
        dataset_val = build_dataset.get_dataset(args.dataset, 'test')
    else:
        dataset_val = build_dataset.get_dataset(args.dataset, 'val')
    dataset_train.set_load_query(False) # during training, we don't need to load the query info
    # dataset_train.set_load_query(True) # during training, we don't need to load the query info

    dataset_train.set_load_radar(True)
    dataset_val.set_load_radar(True)

    # if True:  # args.system.distributed:
    if args.train.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.system.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.system.log_dir is not None and args.system.mode != 'eval':
        os.makedirs(args.system.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.system.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.dataset.batch_size,
        num_workers=args.dataset.num_workers,
        pin_memory=args.dataset.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=args.batch_size,
        batch_size=args.dataset.eval_batch_size,
        num_workers=args.dataset.eval_num_workers,
        # pin_memory=args.dataset.pin_mem,
        pin_memory=False,
        drop_last=False
    )

    if not args.train.use_cache_latent:
        vae:nn.Module = models_ae.__dict__[args.lidar_ae.name](N=args.dataset.lidar.num_samples)
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        vae.load_state_dict(torch.load(args.lidar_ae.ckpt, map_location='cpu')[
                            'model'], strict=True)
        vae.to(device)
        print('Loaded VAE from %s' % args.lidar_ae.ckpt)
    else:
        vae = None
        print('Using cached latents from %s' % args.dataset.cache_latent_sub_dir)

    model = models_radar_generation.__dict__[args.ar_model.name](configs=args.ar_model.configs)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # radar encoder
    if args.ar_model.configs.use_radar_enc and not args.ar_model.configs.get('unfreeze_radar_enc', False):
        print("Using freezed radar encoder")
        radar_enc = models_radar_encoder.__dict__[args.radar_enc.name]()
        for param in radar_enc.parameters():
            param.requires_grad = False
        radar_enc.eval()
        radar_enc.load_state_dict(torch.load(args.radar_enc.ckpt, map_location='cpu')[
                            'model'], strict=True)
        radar_enc.to(device)
        print('Loaded radar encoder from %s' % args.radar_enc.ckpt)
    else:
        radar_enc = None
        print("Not using freezed radar encoder")

    eff_batch_size = args.dataset.batch_size * args.train.accum_iter * misc.get_world_size()
    
    if args.train.lr is None:  # only base_lr is specified
        args.train.lr = float(args.train.blr) * eff_batch_size / 256

    print("base lr: %.2e" % (args.train.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.train.lr)

    print("accumulate grad iterations: %d" % args.train.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.train.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.train.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.train.lr)
    loss_scaler = NativeScaler()

    criterion = models_radar_generation.__dict__['EDMLoss']()

    print("criterion = %s" % str(criterion))
    model_params, ema_params = misc.load_model(args=args, 
                                               model_without_ddp=model_without_ddp, 
                                               optimizer=optimizer, loss_scaler=loss_scaler,
                                               ema=True,device=device)
    if args.system.mode == 'eval':
        if vae is None:
            vae:nn.Module = models_ae.__dict__[args.lidar_ae.name](N=args.dataset.lidar.num_samples)
            for param in vae.parameters():
                param.requires_grad = False
            vae.eval()
            vae.load_state_dict(torch.load(args.lidar_ae.ckpt, map_location='cpu')[
                                'model'], strict=True)
            vae.to(device)
            print('Loaded VAE from %s' % args.lidar_ae.ckpt)
        test_stats = evaluate(data_loader_val, model_without_ddp, vae, device, args=args, 
                              radar_enc=radar_enc)
        print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
        return

    # training
    print(f"Start training for {args.train.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.train.start_epoch, args.train.epochs):
        if args.train.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, vae, criterion,
            model_params, ema_params, 
            data_loader_train, optimizer, 
            device, epoch, loss_scaler,
            args.train.clip_grad,
            log_writer=log_writer,
            args=args,
            radar_enc=radar_enc,
        )

        # save checkpoint
        if args.system.output_dir and (epoch % args.train.save_ckpt_freq == 0 or epoch + 1 == args.train.epochs):
            misc.save_model(
                args=args.system, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, ema_params=ema_params, epoch=epoch)

        # online evaluation
        if epoch % args.train.eval_freq == 0 or epoch + 1 == args.train.epochs:
            if epoch == 0:
                continue
            if vae is None:
                vae:nn.Module = models_ae.__dict__[args.lidar_ae.name](N=args.dataset.lidar.num_samples)
                for param in vae.parameters():
                    param.requires_grad = False
                vae.eval()
                vae.load_state_dict(torch.load(args.lidar_ae.ckpt, map_location='cpu')[
                                    'model'], strict=True)
                vae.to(device)
                print('Loaded VAE from %s' % args.lidar_ae.ckpt)

            test_stats = evaluate(data_loader_val, model_without_ddp, vae, device, args=args, 
                                  radar_enc=radar_enc)
            print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
            max_iou = max(max_iou, test_stats["iou"])
            print(f'Max iou: {max_iou:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
            # free memory of vae
            if args.train.use_cache_latent:
                del vae
                vae = None
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

        # log data
        if args.system.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.system.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    configs = yaml.safe_load(open(args.config))
    configs = edict(configs)
    configs.system.output_dir = os.path.join(configs.system.output_dir, configs.system.expname)
    configs.system.log_dir = os.path.join(configs.system.log_dir, configs.system.expname)
    if configs.dataset.split_file and isinstance(configs.dataset.split_file, dict):
        scene = configs.dataset.split_file
        for key, value in scene.items():
            configs.dataset.split_file = value
            configs.system.output_dir = os.path.join(configs.system.output_dir, key)
            configs.system.log_dir = os.path.join(configs.system.log_dir, key)
            if configs.system.output_dir:
                ensure_path_exists(Path(configs.system.output_dir))
                with open(os.path.join(configs.system.output_dir, Path(args.config).name), 'w') as f:
                    yaml.dump(configs, f)
            main(configs)
    else:
        if configs.system.output_dir:
            ensure_path_exists(Path(configs.system.output_dir))
            with open(os.path.join(configs.system.output_dir, Path(args.config).name), 'w') as f:
                yaml.dump(configs, f)
        main(configs)
            
