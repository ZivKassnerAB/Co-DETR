#!/bin/bash

OUTPUT_PARENT_DIR="/home/ubuntu/workspace/results/ab_car_validation_germany/faster_rcnn_cityscapes"
OUTPUT_INFERENCE_DIR="${OUTPUT_PARENT_DIR}/infer_no_scale"

mkdir -p ${OUTPUT_INFERENCE_DIR}

python sahi_infer.py \
--config_path /home/ubuntu/git/Co-DETR/main_config.ini \
--images_dir /home/ubuntu/workspace/datasets/ab_car_validation_germany/images \
--output_dir ${OUTPUT_INFERENCE_DIR} \
> ${OUTPUT_INFERENCE_DIR}/log.txt 2>&1
