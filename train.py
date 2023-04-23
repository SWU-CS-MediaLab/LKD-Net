import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging
import time
import shutil
import torch.backends.cudnn as cudnn
from utils import AverageMeter
from datasets.loader import PairLoader
from utils.utils import create_logger, summary_model, \
    save_checkpoint, resume_checkpoint, save_model, \
    set_seed_torch
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='LKD-t', type=str, help='model')
parser.add_argument('--model_name', default='LKD.py', type=str, help='model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./result', type=str,
                    help='path to models saving')
parser.add_argument('--resume_checkpoint', default=True, type=bool,
                    help='resume checkpoint')

# dataset config
parser.add_argument('--datasets_dir', default='./data', type=str, help='path to datasets dir')
parser.add_argument('--train_dataset', default='ITS', type=str, help='train dataset name')
parser.add_argument('--valid_dataset', default='SOTS', type=str, help='valid dataset name')
parser.add_argument('--exp_config', default='indoor', type=str, help='experiment configuration')
parser.add_argument('--exp_name', default='test', type=str, help='experiment name')

parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

parser.add_argument('--cudnn_BENCHMARK', default=True)
parser.add_argument('--cudnn_DETERMINISTIC', default=False)
parser.add_argument('--cudnn_ENABLED', default=True)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    torch.cuda.empty_cache()

    network.train()
    pbar = tqdm(desc="Epoch[{0}]".format(epoch), total=len(train_loader), leave=True,
                ncols=160)
    end = time.time()
    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):
            output = network(source_img)
            loss1 = criterion[0](output, target_img)
            # loss2 = criterion[1](output, target_img)
            loss = loss1

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(Speed="{:.1f} samples/s".format(output.size(0) / batch_time.val),
                         Loss="{:.5f}".format(loss))
        pbar.update()
    pbar.close()
    return losses.avg

def valid(val_loader, network):
    losses = AverageMeter()
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()
    end = time.time()
    # init progress bar

    pbar = tqdm(desc="Testing", total=len(val_loader), leave=True, ncols=160)
    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img)
            loss1 = criterion[0](output, target_img)
            # loss2 = criterion[1](output, target_img)
            loss = loss1

        losses.update(loss.item())

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        # mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        pbar.set_postfix(PSNR="{:.2f}db".format(psnr))
        pbar.update()
        PSNR.update(psnr.item(), source_img.size(0))
    pbar.close()
    return losses.avg, PSNR.avg


def setup_cudnn(config):
    cudnn.benchmark = config.cudnn_BENCHMARK
    torch.backends.cudnn.deterministic = config.cudnn_DETERMINISTIC
    torch.backends.cudnn.enabled = config.cudnn_ENABLED



if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp_config, args.model + '.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # set random seed
    set_seed_torch()

    setup_cudnn(args)

    # Create logger
    final_output_dir = os.path.join(args.save_dir, args.train_dataset, args.exp_name)
    create_logger(final_output_dir)

    # build network
    network = eval(args.model.replace('-', '_'))()

    # copy config file
    shutil.copy2(
        setting_filename,
        final_output_dir
    )

    # copy model file
    summary_model(network, args.model_name, final_output_dir, [256, 256])

    network = nn.DataParallel(network).cuda()

    # build criterion
    criterion = []
    criterion.append(nn.L1Loss().cuda())

    # build optimizer
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    # resume checkpoint
    best_psnr, begin_epoch = resume_checkpoint(network, optimizer, args, final_output_dir, True)

    # build scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2, last_epoch=begin_epoch - 1)
    # build scaler
    scaler = GradScaler()

    # build dataloader
    train_dataset = PairLoader(args.datasets_dir, args.train_dataset, 'train',
                               setting['patch_size'], setting['only_h_flip'])
    val_dataset = PairLoader(args.datasets_dir, os.path.join(args.valid_dataset, args.exp_config), 'valid',
                                setting['valid_mode'], setting['patch_size'])

    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)
    # init SummaryWriter
    writer = SummaryWriter(log_dir=final_output_dir)

    # begin epoch
    logging.info('=> start training')

    for epoch in range(begin_epoch, setting['epochs'] + 1):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} train start'.format(head))
        lr = scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        start = time.time()
        train_loss = train(train_loader, network, criterion, optimizer, scaler)
        writer.add_scalars('Loss', {'train Loss': train_loss}, epoch)
        msg = '=> Train:\t' \
              'Loss {:.4f}\t'.format(train_loss)
        logging.info(msg)
        logging.info('=> {} train end, duration: {:.2f}s'.format(head, time.time() - start))

        scheduler.step(epoch=epoch + 1)

        save_checkpoint(model=network, model_name=args.model.replace('-', '_'), optimizer=optimizer,
                        output_dir=final_output_dir, in_epoch=True, epoch_or_step=epoch, best_perf=best_psnr)

        if epoch % setting['eval_freq'] == 0:
            logging.info('=> {} validate start'.format(head))

            val_start = time.time()
            valid_loss, avg_psnr = valid(val_loader, network)
            writer.add_scalars('Loss', {'valid Loss': valid_loss}, epoch)
            msg = '=> Valid:\t' \
                  'Loss {:.4f}\t' \
                  'PSNR {:.2f}\t'.format(valid_loss, avg_psnr)
            logging.info(msg)

            logging.info('=> {} validate end, duration: {:.2f}s'.format(head, time.time() - val_start))
            writer.add_scalar('valid_psnr', avg_psnr, epoch)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_model(network, final_output_dir, 'best_model.pth')
                writer.add_scalar('best_psnr', best_psnr, epoch)

    writer.close()
    save_model(network, final_output_dir, 'final_model.pth')
    logging.info('=> finish training')
    logging.info("=> Highest PSNR:{:.2f}".format(best_psnr))
