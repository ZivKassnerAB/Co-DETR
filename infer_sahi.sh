#!/bin/bash

python sahi_infer.py \
--config_path /home/ubuntu/git/Co-DETR/main_config.ini \
--images_dir /home/ubuntu/workspace/datasets/ab_car_validation_germany/images \
--output_dir /home/ubuntu/workspace/results/ab_car_validation_germany/faster_rcnn_cityscapes/infer \
> /home/ubuntu/workspace/results/ab_car_validation_germany/faster_rcnn_cityscapes/infer/log.txt 2>&1