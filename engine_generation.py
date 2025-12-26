# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import copy
import numpy as np
from pathlib import Path
import open3d as o3d

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.utils import inverse_norm_points, norm_points, remove_points_outside_fov,\
                        generate_query_points,cal_metrics, ensure_path_exists
from dataset_preprocessor.lidar import polar2cartesian,cartesian2polar
from datasets.utils.query_helper import aug_query_helper

from model.models_ae import  KLAutoEncoder
from model.models_radar_encoder import RadarAutoencoder
from model.models_radar_generation import EDMPrecond, EDMLoss

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model: EDMPrecond,  vae: KLAutoEncoder, criterion: EDMLoss,
                    model_params, ema_params, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, radar_enc:RadarAutoencoder|None =None, 
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.train.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (data_dict)  in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args.train)

        pc = data_dict['lidar_points']
        radar_cube = data_dict['radar_cube'] if args.ar_model.configs.get("use_radar_cond", True) else None
        pc = pc.to(device, non_blocking=True)
        if radar_cube is not None:
            radar_cube = radar_cube.to(device, non_blocking=True)


        with torch.no_grad():
            if args.train.use_cache_latent:
                # check the key 'res_latens' in data_dict
                assert 'cache_latent' in data_dict, "The key 'res_latents' is not in data_dict"
                res_latents = data_dict['cache_latent'].to(device, non_blocking=True)
            else:
                if isinstance(vae, KLAutoEncoder):
                    kl, res_latents  = vae.encode(pc)
                else:
                    raise NotImplementedError(f"VAE type {type(vae)} is not supported")
                
            if args.ar_model.configs.use_radar_enc and (not args.ar_model.configs.get('unfreeze_radar_enc', False)):
                assert radar_enc is not None, "The radar encoder is not provided"
                radar_cube = radar_enc._encode(radar_cube)          

        # forward pass
        cond_type = args.ar_model.configs.cond_type
        with torch.amp.autocast('cuda',enabled=False):
            if cond_type == 'radar':
                loss = criterion(model, res_latents, radar_cube, cond_type)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        norm=loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=0.999)

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            log_writer.add_scalar('norm', norm, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model:EDMPrecond, vae: KLAutoEncoder, device,
             args=None, radar_enc:RadarAutoencoder|None=None, ema_params=None):
    criterion_BCE = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    use_ema = args.train.get("use_ema", False)
    if use_ema:
        assert ema_params is not None, "EMA parameters are not provided"
        print("Using EMA parameters for evaluation")

        # switch model parameters to EMA parameters
        model_state_dict = copy.deepcopy(model.state_dict())
        ema_state_dict = copy.deepcopy(model.state_dict())
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in ema_params, f"Parameter {name} is not in ema_params"
            ema_state_dict[name] = ema_params[i]
        print("Loaded EMA parameters into the model")
        model.load_state_dict(ema_state_dict, strict=True)
    else:
        print("Using model parameters for evaluation")

    eval_freq = args.eval.freq if hasattr(args.eval, 'freq') else 1

    for data_iter_step, (data_dict) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        if data_iter_step % eval_freq != 0:
            continue
        points = data_dict['query_points']
        labels = data_dict['query_labels']
        surface = data_dict['lidar_points']
        radar_cube = data_dict['radar_cube'] if args.ar_model.configs.get("use_radar_cond", True) else None
        
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        if radar_cube is not None:
            radar_cube = radar_cube.to(device, non_blocking=True)
        surface_np = surface.cpu().numpy()

        # compute output
        with torch.amp.autocast('cuda',enabled=False):     
            if args.eval.get('use_pred_latent', False):    
                sampled_tokens = data_dict['pred_latent'].to(device, non_blocking=True).squeeze(1)  # [B, M, D]

            # get the sampled tokens from the model
            else:
                if args.ar_model.configs.use_radar_enc and (not args.ar_model.configs.get('unfreeze_radar_enc', False)):
                    assert radar_enc is not None, "The radar encoder is not provided"
                    radar_cube = radar_enc._encode(radar_cube)  

                cond_type = args.ar_model.configs.cond_type
                if cond_type == 'radar':
                    sampled_tokens = model.sample(cond=radar_cube, batch_seeds=None,cond_type=cond_type).to(torch.float32)   

            if args.eval.get('iou_test_only', False):
                if isinstance(vae, KLAutoEncoder):
                    outputs = vae.decode(sampled_tokens, surface).squeeze(-1)
                else:
                    raise NotImplementedError(f"VAE type {type(vae)} is not supported")    
            else:
                if isinstance(vae, KLAutoEncoder):
                    outputs = vae.decode(sampled_tokens, points).squeeze(-1)
                else:
                    raise NotImplementedError(f"VAE type {type(vae)} is not supported")   


            if args.eval.get('store_latent', False):
                # store the latent tokens
                sampled_tokens_cpu = sampled_tokens.cpu()
                base_dir = args.eval.store_base_dir
                B = sampled_tokens.shape[0]
                for i in range(B):
                    seq_name = Path(data_dict['lidar_path'][0][i]).parent.parent.name
                    save_dir = Path(base_dir) / args.eval.exp_name / seq_name / 'latent_tokens'
                    ensure_path_exists(save_dir)
                    if data_iter_step == 0:
                        print(f'Caching generated tokens at {save_dir}')
                    data_idx = Path(data_dict['radar_path'][0][i]).stem + ".pt"
                    save_path = save_dir / data_idx
                    torch.save(sampled_tokens_cpu, save_path)

            if args.eval.get('test_sample_speed', False):
                loss = torch.tensor(-1.0)
            else:
                loss = criterion_BCE(outputs, labels)
                            
        threshold = 0

        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        # skip accuracy and iou calculation when only testing the sampling speed
        if args.eval.get('test_sample_speed', False):
            iou=torch.tensor(-1.0)
        else:
            accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
            accuracy = accuracy.mean()
            intersection = (pred * labels).sum(dim=1)
            union = (pred + labels).gt(0).sum(dim=1)
            iou = intersection * 1.0 / union + 1e-5
            iou = iou.mean()

        batch_size = points.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['iou'].update(iou.item(), n=batch_size)

        if not args.eval.get('iou_test_only', False):
            # generate random query points
            if args.eval.get('use_cart_query', False):
                grid_np_cart = generate_query_points(args,coordinate_type='cart')
                grid_np_cart = inverse_norm_points(grid_np_cart, args.dataset.lidar.pc_range_cart, args.dataset.lidar.norm_anisotropy, args.dataset.lidar.norm_isotropy)
                grid_np = cartesian2polar(grid_np_cart)
                grid_np = norm_points(grid_np, args.dataset.lidar.pc_range,args.dataset.lidar.norm_anisotropy, args.dataset.lidar.norm_isotropy)
                grid_np = remove_points_outside_fov(grid_np)
            else:
                grid_np = generate_query_points(args)
            grid_np = grid_np.repeat(batch_size, axis=0).reshape(batch_size, -1, 3).astype('float32')
            grid = torch.from_numpy(grid_np).float()
            grid = grid.to(device, non_blocking=True)  
            grid = grid.type(torch.float32)   

            if args.eval.inference.get('query_helper', False):
                assert batch_size == 1, "Batch size should be 1 when using query helper points"
                # use the helper points to generate the query points
                helper_points = data_dict['helper_points']  # [B, H, 3]
                helper_points_np = helper_points.numpy()
                helper_points = helper_points.to(device, non_blocking=True)
                grid = torch.cat((grid, helper_points), dim=1)  # [B, M+H, 3]
                grid_np = np.concat((grid_np, helper_points_np), axis=1)
                del helper_points

            if isinstance(vae, KLAutoEncoder):
                output = vae.decode(sampled_tokens, grid).squeeze(-1)
            else:
                raise NotImplementedError(f"VAE type {type(vae)} is not supported")   
            if args.eval.get('test_sample_speed', False):
                continue
            del grid
            cd_all=[]
            for i in range(batch_size):
                output_np = output[i].cpu().numpy()
                # del output
                ind_pos = np.where(output_np > 0)[0]
                
                # Inverse normalization
                grid_pos = grid_np[i][ind_pos] 
                pred = inverse_norm_points(grid_pos, args.dataset.lidar.pc_range, args.dataset.lidar.norm_anisotropy, args.dataset.lidar.norm_isotropy)
                ground_truth = inverse_norm_points(surface_np[i], args.dataset.lidar.pc_range, args.dataset.lidar.norm_anisotropy, args.dataset.lidar.norm_isotropy)
                
                if args.eval.inference.get('refine_query', False):
                    refined_query_np = aug_query_helper(pred, int(args.eval.inference.refine_query_aug_num),
                                                    args.dataset.lidar.pc_range, args.dataset.lidar.voxel_size,
                                                    args.eval.inference.refine_query_scale)                     #[M, 3]
                    refined_query_np = norm_points(refined_query_np, args.dataset.lidar.pc_range,
                                                args.dataset.lidar.norm_anisotropy, args.dataset.lidar.norm_isotropy)   # [M, 3]
                    refined_query = torch.from_numpy(refined_query_np).float().to(device, non_blocking=True).unsqueeze(0)  # [M, 3]
                    if isinstance(vae, KLAutoEncoder):
                        output_refined = vae.decode(sampled_tokens, refined_query).squeeze(-1)
                    else:
                        raise NotImplementedError(f"VAE type {type(vae)} is not supported")

                    output_np_refined = output_refined[0].cpu().numpy()
                    ind_pos_refined = np.where(output_np_refined > 0)[0]
                    
                    # Inverse normalization
                    grid_pos_refined = refined_query_np[ind_pos_refined]
                    pred = inverse_norm_points(grid_pos_refined, args.dataset.lidar.pc_range, 
                                               args.dataset.lidar.norm_anisotropy, args.dataset.lidar.norm_isotropy)

                # if is view cone mode, transform point to cartesian coord first
                if args.dataset.lidar.get('view_cone_mode', False):
                    pred = polar2cartesian(pred)
                    ground_truth = polar2cartesian(ground_truth)

                if args.eval.get("skip_eval_metric", False):
                    cd = -1.0
                else:
                    # calulate metrics
                    cd = cal_metrics(y_pred=pred, y_gt=ground_truth)
                cd_all.append(cd)

                if args.eval.get('store_pc', False):
                    # store the latent tokens
                    base_dir = args.eval.store_base_dir
                    B = sampled_tokens.shape[0]
                    for i in range(B):
                        seq_name = Path(data_dict['radar_path'][0][i]).parent.parent.parent.name
                        save_dir = Path(base_dir) / args.eval.exp_name / seq_name / args.eval.save_pc_dir_name
                        ensure_path_exists(save_dir)
                        if data_iter_step == 0:
                            print(f'Caching generated pc at {save_dir}')
                        data_idx = Path(data_dict['radar_path'][0][i]).stem + ".ply"
                        save_path = save_dir / data_idx
                        pred_o3d = o3d.geometry.PointCloud()
                        pred_o3d.points = o3d.utility.Vector3dVector(pred)
                        o3d.io.write_point_cloud(str(save_path), pred_o3d)                    

            metric_logger.meters['cd'].update(np.mean(cd), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* iou {iou.global_avg:.3f} loss {losses.global_avg:.3f} ' \
            'cd {cd.global_avg:.3f} ' \
            # 'loss_latent_token {loss_latent_token.global_avg:.3f} ' \
          .format(iou=metric_logger.iou, losses=metric_logger.loss, 
                  cd=metric_logger.cd))

    # switch back to origin model from ema
    if use_ema:
        print('Switch back from ema')
        model.load_state_dict(model_state_dict)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def cache_latents(vae: KLAutoEncoder, data_loader, device, cache_base_path):
    """
    Cache the latents of the training data.
    """
    import os
    from pathlib import Path
    from utils.utils import ensure_path_exists
    import numpy as np

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 50

    vae.eval()
    for i, (data_dict) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        pcs = data_dict['lidar_points']
        points = data_dict['query_points']
        labels = data_dict['query_labels']
        lidar_paths:list[Path] = [Path(x) for x in data_dict['lidar_path'][0]]


        pcs = pcs.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = pcs.shape[0]
        with torch.no_grad():
            if isinstance(vae, KLAutoEncoder):
                kl, res_latents  = vae.encode(pcs)
                outputs = vae.decode(res_latents, points).squeeze(-1)
            else:
                raise NotImplementedError(f"VAE type {type(vae)} is not supported")
            
        threshold = 0
        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        metric_logger.meters['iou'].update(iou.item(), n=B)

        for idx in range(B):
            seq_name, frame_name = lidar_paths[idx].parts[-3], lidar_paths[idx].parts[-1]
            # get the path of the latent
            latens_dir = cache_base_path / seq_name
            ensure_path_exists(latens_dir)
            latens_path = latens_dir / (frame_name + '.npz')
            # save the latens
            np.savez(latens_path, res_tokens=res_latents[idx].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

 