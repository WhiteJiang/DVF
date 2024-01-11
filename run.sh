#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_dp.py --dataset cub  --config config/vit.json --mode trainval --seed 42