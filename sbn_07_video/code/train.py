import os
import math
import argparse
import random
import logging
import csv
import json
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def cal_pnsr(sr_img, gt_img):
    # calculate PSNR
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.
    psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)

    return psnr


def cal_ssim(sr_img, gt_img):
    # calculate SSIM
    gt_img = gt_img / 255.
    sr_img = sr_img / 255.
    ssim = util.calculate_ssim(sr_img * 255, gt_img * 255)

    return ssim


def format_metric_list(values):
    return ''.join('_{:.4e}'.format(v) for v in values)


def format_eta(seconds):
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)


def save_metrics(rows, summary, csv_path, json_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')  # config 文件
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        allow_tf32 = bool(opt['train'].get('allow_tf32', True))
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except AttributeError:
            pass
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)
    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    time_ref = time.time()
    step_ref = current_step

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            # update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                done_steps = max(1, current_step - step_ref)
                elapsed = max(1e-6, time.time() - time_ref)
                avg_step_time = elapsed / done_steps
                eta = avg_step_time * max(0, total_iters - current_step)
                progress = 100.0 * float(current_step) / float(total_iters)
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                message += 'progress: {:.2f}% time/iter: {:.3f}s eta: {} '.format(
                    progress, avg_step_time, format_eta(eta)
                )
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_psnr_h = None
                avg_psnr_lr = 0.0
                avg_ssim = 0.0
                avg_ssim_h = None
                avg_ssim_lr = 0.0
                idx = 0
                metrics_rows = []
                for video_id, val_data in enumerate(val_loader):
                    img_dir = os.path.join(opt['path']['val_images'])
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()

                    t_step = visuals['SR'].shape[0]
                    idx += t_step
                    n = len(visuals['SR_h'])
                    if avg_psnr_h is None:
                        avg_psnr_h = [0.0] * n
                        avg_ssim_h = [0.0] * n

                    for i in range(t_step):

                        sr_img = util.tensor2img(visuals['SR'][i])  # uint8
                        sr_img_h = []
                        for j in range(n):
                            sr_img_h.append(util.tensor2img(visuals['SR_h'][j][i]))  # uint8
                        gt_img = util.tensor2img(visuals['GT'][i])  # uint8
                        lr_img = util.tensor2img(visuals['LR'][i])
                        lrgt_img = []
                        for j in range(n):
                            lrgt_img.append(util.tensor2img(visuals['LR_ref'][j][i]))

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'SR'))
                        util.save_img(sr_img, save_img_path)

                        for j in range(n):
                            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(video_id, i, j, 'SR_h'))
                            util.save_img(sr_img_h[j], save_img_path)

                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'GT'))
                        util.save_img(gt_img, save_img_path)

                        save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:s}.png'.format(video_id, i, 'LR'))
                        util.save_img(lr_img, save_img_path)

                        for j in range(n):
                            save_img_path = os.path.join(img_dir,'{:d}_{:d}_{:d}_{:s}.png'.format(video_id, i, j, 'LRGT'))
                            util.save_img(lrgt_img[j], save_img_path)

                        psnr = cal_pnsr(sr_img, gt_img)
                        ssim = cal_ssim(sr_img, gt_img)
                        psnr_h = []
                        ssim_h = []
                        for j in range(n):
                            psnr_h.append(cal_pnsr(sr_img_h[j], lrgt_img[j]))
                            ssim_h.append(cal_ssim(sr_img_h[j], lrgt_img[j]))
                        psnr_lr = cal_pnsr(lr_img, gt_img)
                        ssim_lr = cal_ssim(lr_img, gt_img)

                        avg_psnr += psnr
                        avg_ssim += ssim
                        for j in range(n):
                            avg_psnr_h[j] += psnr_h[j]
                            avg_ssim_h[j] += ssim_h[j]
                        avg_psnr_lr += psnr_lr
                        avg_ssim_lr += ssim_lr

                        row = {
                            'video_id': int(video_id),
                            'frame_id': int(i),
                            'psnr_cover': float(psnr),
                            'ssim_cover': float(ssim),
                            'psnr_stego': float(psnr_lr),
                            'ssim_stego': float(ssim_lr),
                        }
                        for j in range(n):
                            row['psnr_secret_{}'.format(j)] = float(psnr_h[j])
                            row['ssim_secret_{}'.format(j)] = float(ssim_h[j])
                        metrics_rows.append(row)

                avg_psnr = avg_psnr / idx
                avg_psnr_h = [psnr / idx for psnr in avg_psnr_h]
                avg_psnr_lr = avg_psnr_lr / idx
                avg_ssim = avg_ssim / idx
                avg_ssim_h = [ssim / idx for ssim in avg_ssim_h]
                avg_ssim_lr = avg_ssim_lr / idx

                # log
                res_psnr_h = format_metric_list(avg_psnr_h)
                res_ssim_h = format_metric_list(avg_ssim_h)

                logger.info('# Validation # PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, PSNR_Stego: {:.4e}; SSIM_Cover: {:.4e}, SSIM_Secret: {:s}, SSIM_Stego: {:.4e}'.format(
                    avg_psnr, res_psnr_h, avg_psnr_lr, avg_ssim, res_ssim_h, avg_ssim_lr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> PSNR_Cover: {:.4e}, PSNR_Secret: {:s}, PSNR_Stego: {:.4e}; SSIM_Cover: {:.4e}, SSIM_Secret: {:s}, SSIM_Stego: {:.4e}'.format(
                    epoch, current_step, avg_psnr, res_psnr_h, avg_psnr_lr, avg_ssim, res_ssim_h, avg_ssim_lr))

                metrics_dir = os.path.join(opt['path']['experiments_root'], 'metrics')
                metrics_csv_path = os.path.join(metrics_dir, 'val_iter_{:08d}.csv'.format(current_step))
                metrics_json_path = os.path.join(metrics_dir, 'val_iter_{:08d}_summary.json'.format(current_step))
                summary = {
                    'epoch': int(epoch),
                    'iter': int(current_step),
                    'psnr_cover': float(avg_psnr),
                    'psnr_secret': [float(v) for v in avg_psnr_h],
                    'psnr_stego': float(avg_psnr_lr),
                    'ssim_cover': float(avg_ssim),
                    'ssim_secret': [float(v) for v in avg_ssim_h],
                    'ssim_stego': float(avg_ssim_lr),
                }
                save_metrics(metrics_rows, summary, metrics_csv_path, metrics_json_path)
                logger.info('Saved validation metrics to {} and {}'.format(metrics_csv_path, metrics_json_path))

                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
