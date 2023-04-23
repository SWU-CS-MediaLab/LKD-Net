from pathlib import Path

import os
import logging
import shutil
import time
import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis
import numpy as np


def setup_logger(final_output_dir, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.txt'.format(phase, time_str)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:[P:%(process)d]:' + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)


def create_logger(final_output_dir, phase='train'):
    final_output_dir = Path(final_output_dir)
    print('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> setup logger ...')
    setup_logger(final_output_dir, phase)


def set_seed_torch(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def summary_model(model, model_name, output_dir, image_size=(256, 256)):
    this_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # copy model file
    shutil.copy2(
        os.path.join(this_dir, 'models', model_name),
        output_dir
    )
    try:
        logging.info('== get_model_complexity_info by thop and fvcore ==')
        input = torch.randn(1, 3, image_size[0], image_size[1])
        flops = FlopCountAnalysis(model, input)
        _, params = profile(model, inputs=(input,))
        flops = flops.total() / 1e9
        params = params / 1e6
        logging.info(f'=> FLOPs: {flops:<8}G, params: {params:<8}M')
        logging.info('== get_model_complexity_info by thop and fvcore ==')
    except Exception:
        logging.error('=> error when run get_model_complexity_info')


def resume_checkpoint(model,
                      optimizer,
                      config,
                      output_dir,
                      in_epoch):
    best_perf = 0.0
    begin_epoch_or_step = 0

    checkpoint = os.path.join(output_dir, 'checkpoint.pth')
    if config.resume_checkpoint and os.path.exists(checkpoint):
        logging.info(
            "=> loading checkpoint '{}'".format(checkpoint)
        )
        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        best_perf = checkpoint_dict['perf']
        begin_epoch_or_step = checkpoint_dict['epoch' if in_epoch else 'step']
        state_dict = checkpoint_dict['state_dict']
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        logging.info(
            "=> loaded checkpoint '{}' ({}: {})".format(checkpoint, 'epoch' if in_epoch else 'step',
                                                        begin_epoch_or_step)
        )

    return best_perf, begin_epoch_or_step


def save_checkpoint(model,
                    *,
                    model_name,
                    optimizer,
                    output_dir,
                    in_epoch,
                    epoch_or_step,
                    best_perf):
    states = model.state_dict()

    logging.info('=> saving checkpoint to {}'.format(output_dir))
    save_dict = {
        'epoch' if in_epoch else 'step': epoch_or_step + 1,
        'model': model_name,
        'state_dict': states,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
    }

    try:
        torch.save(save_dict, os.path.join(output_dir, 'checkpoint.pth'))
    except Exception:
        logging.error('=> error when saving checkpoint!')


def save_model(model, out_dir, fname):
    try:
        fname_full = os.path.join(out_dir, fname)
        logging.info(f'=> save model to {fname_full}')
        torch.save(
            model.state_dict(),
            fname_full
        )
    except Exception:
        logging.error('=> error when saving checkpoint!')
