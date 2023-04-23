import os
import argparse
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='LKD-s', type=str, help='model name')
parser.add_argument('--model_weight', default='./result/RESIDE-OUT/LKD-s/LKD-s.pth', type=str,
                    help='model weight file name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--datasets_dir', default='./data', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./result',
                    type=str, help='path to models saving')
parser.add_argument('--dataset', default='SOTS', type=str, help='dataset name')
parser.add_argument('--subset', default='outdoor', type=str, help='subset')
parser.add_argument('--mode', default='valid', type=str, help='dataset mode')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def single(save_dir):
    state_dict = torch.load(save_dir, map_location=torch.device(device))
    # print(state_dict)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].to(device)
        target = batch['target'].to(device)

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.write('Avg_all,%.02f,%.03f\n' % (PSNR.avg, SSIM.avg))
    f_result.close()


if __name__ == '__main__':

    # load model
    network = eval(args.model.replace('-', '_'))()
    network.to(device)

    # load pre-trained weight
    if os.path.exists(args.model_weight):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(args.model_weight), strict=False)
    else:
        print('==> No existing trained model!')
        exit(0)

    # load dataset
    test_dataset = PairLoader(args.datasets_dir, os.path.join(args.dataset, args.subset), args.mode)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.save_dir, 'test_result', args.model, args.dataset, args.subset)

    # begin test
    test(test_loader, network, result_dir)
