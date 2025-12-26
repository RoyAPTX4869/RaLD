# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import torch
import copy
import numpy as np
import utils.misc as misc
import utils.lr_sched as lr_sched
from model.models_ae import  KLAutoEncoder
from utils.utils import inverse_norm_points, norm_points, remove_points_outside_fov,\
                        generate_query_points,cal_metrics
from dataset_preprocessor.lidar import polar2cartesian,cartesian2polar

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

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.train.accum_iter

    optimizer.zero_grad()

    kl_weight = 1e-3
    near_weight = args.train.near_weight 
    vol_weight = args.train.vol_weight

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (data_dict) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args.train)
        points = data_dict['query_points']
        labels = data_dict['query_labels']
        surface = data_dict['lidar_points']
        in_voxel_num = data_dict['in_voxel_num'][0]
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        with torch.amp.autocast('cuda',enabled=False):
            outputs = model(surface, points)
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']

            
            loss_vol = criterion(outputs[:, :in_voxel_num], labels[:, :in_voxel_num])
            loss_near = criterion(outputs[:, in_voxel_num:], labels[:, in_voxel_num:])

            
            if loss_kl is not None:
                loss = vol_weight*loss_vol + near_weight * loss_near + kl_weight * loss_kl
            else:
                loss = vol_weight*loss_vol + near_weight * loss_near

        loss_value = loss.item()

        threshold = 0

        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1


        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1) + 1e-5
        iou = intersection * 1.0 / union
        iou = iou.mean()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        update_ema(ema_params, model_params, rate=0.999)

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        metric_logger.update(iou=iou.item())

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
            log_writer.add_scalar('vol_loss', loss_vol.item(), epoch_1000x)
            log_writer.add_scalar('near_loss', loss_near.item(), epoch_1000x)
            if loss_kl is not None:
                log_writer.add_scalar('kl_loss', loss_kl.item(), epoch_1000x)
            log_writer.add_scalar('iou', iou.item(), epoch_1000x)
            log_writer.add_scalar('accuracy', accuracy.item(), epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model: KLAutoEncoder, device,args=None,ema_params=None):
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

    for data_iter_step, (data_dict) in enumerate(metric_logger.log_every(data_loader, 50, header)):
        if data_iter_step % eval_freq != 0:
            continue
        points = data_dict['query_points']
        labels = data_dict['query_labels']
        surface = data_dict['lidar_points']
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)

        surface_np = surface.cpu().numpy()
        
        # compute output
        with torch.amp.autocast('cuda',enabled=False): 
            if isinstance(model, KLAutoEncoder):
                outputs = model(surface, points)
            else:
                raise NotImplementedError(f"VAE type {type(model)} is not supported") 
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']

            loss = criterion_BCE(outputs, labels)

        threshold = 0

        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        batch_size = points.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['iou'].update(iou.item(), n=batch_size)

        if not args.eval.get('iou_test_onlytest', False):
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


            if isinstance(model, KLAutoEncoder):
                output = model(surface, grid)['logits']
            else:
                raise NotImplementedError(f"VAE type {type(vae)} is not supported")   
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