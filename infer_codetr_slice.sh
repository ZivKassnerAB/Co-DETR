#!/bin/bash

python sahi_infer.py \
--images_dir /home/ubuntu/workspace/datasets/ab_car_validation_germany/images \
--output_dir /home/ubuntu/workspace/results/ab_car_validation_germany/riders/Co-DETR/all \
--config_path /home/ubuntu/git/Co-DETR/projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py \
--weights_path /home/ubuntu/git/Co-DETR/co_dino_5scale_lsj_swin_large_3x_coco.pth \
> log.txt 2>&1