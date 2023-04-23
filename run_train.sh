#!/bin/bash

python train.py \
--model LKD-t \
--model_name LKD.py \
--num_workers 8 \
--save_dir ./result \
--datasets_dir ./data \
--train_dataset OTS \
--valid_dataset SOTS \
--exp_config outdoor \
--gpu 0 \
--exp_name test \
