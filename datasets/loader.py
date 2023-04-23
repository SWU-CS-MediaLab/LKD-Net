import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
import torchvision.transforms as tfs


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs


class PairLoader(Dataset):
    def __init__(self, data_dir, dataset_name, mode, size=256, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']
        
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.size = size
        self.edge_decay = 0
        self.only_h_flip = only_h_flip

        self.img_names = sorted(os.listdir(os.path.join(self.data_dir, dataset_name, 'hazy')))
        # read exclude files
        exclude_file = os.path.abspath(os.path.join('datasets','exclude_files', self.dataset_name + '_exclude_file.txt')) 
        if os.path.exists(exclude_file):
            with open(exclude_file, 'r') as f:
                exclude_filenames = eval(f.read())
            # filter out exclude files
            for exclude_filename in exclude_filenames:
                if exclude_filename in self.img_names:
                    self.img_names.remove(exclude_filename)
            
        self.gt_names = sorted(os.listdir(os.path.join(self.data_dir, dataset_name, 'gt')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        hazy_name = self.img_names[idx]


        if self.dataset_name == 'ITS':
            gt_name = hazy_name.split('_')[0] + '.png'
        elif self.dataset_name == 'OTS':
            gt_name = hazy_name.split('_')[0] + '.jpg'
        elif self.dataset_name == 'SOTS/indoor' or self.dataset_name == 'SOTS/outdoor':
            gt_name = hazy_name.split('_')[0] + '.png'
        else:
            gt_name = self.gt_names[idx]

        source_img = read_img(os.path.join(self.data_dir, self.dataset_name, 'hazy', hazy_name))
        target_img = read_img(os.path.join(self.data_dir, self.dataset_name, 'gt', gt_name))

        # scale [0, 1] to [-1, 1]
        source_img = source_img * 2 - 1
        target_img = target_img * 2 - 1

        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': hazy_name}

