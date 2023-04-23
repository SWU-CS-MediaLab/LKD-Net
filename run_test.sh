#!/bin/bash

python test.py --model LKD-t --model_weight ./result/ITS/LKD-t/LKD-t.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset indoor
python test.py --model LKD-s --model_weight ./result/ITS/LKD-s/LKD-s.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset indoor
python test.py --model LKD-b --model_weight ./result/ITS/LKD-b/LKD-b.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset indoor
python test.py --model LKD-l --model_weight ./result/ITS/LKD-l/LKD-l.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset indoor

python test.py --model LKD-t --model_weight ./result/OTS/LKD-t/LKD-t.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset outdoor
python test.py --model LKD-s --model_weight ./result/OTS/LKD-s/LKD-s.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset outdoor
python test.py --model LKD-b --model_weight ./result/OTS/LKD-b/LKD-b.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset outdoor
python test.py --model LKD-l --model_weight ./result/OTS/LKD-l/LKD-l.pth --datasets_dir ./data --save_dir ./result --dataset SOTS --subset outdoor
