import argparse
import configparser
import glob
import os
import pickle
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
import torchvision
import numpy as np
import cv2
import pandas as pd
import torch
from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

import mmcv_custom
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NMSProcessor:
    def __init__(self, iou_thresh=0.5):
        self.iou_thresh = iou_thresh

    def convert_to_corners(self, boxes):
        """ Convert boxes from center format to corners format """
        x_center, y_center, width, height = boxes.T
        x1 = x_center - (width / 2)
        y1 = y_center - (height / 2)
        x2 = x_center + (width / 2)
        y2 = y_center + (height / 2)
        return np.stack((x1, y1, x2, y2), axis=1)

    def nms(self, boxes, scores, labels):
        boxes_corners = self.convert_to_corners(boxes)
        idxs = torchvision.ops.nms(torch.Tensor(boxes_corners), torch.Tensor(scores), iou_threshold=self.iou_thresh).numpy()
        return boxes[idxs], scores[idxs], labels[idxs]

    def nms_by_category(self, df):
        if df.empty:
            return df
        
        boxes = df[['x_center', 'y_center', 'width', 'height']].values
        scores = df['score'].values
        labels = df['label'].values
        
        boxes_list, scores_list, labels_list = [], [], []
        
        for clz in [0, 1, 2, 4]:
            mask = (labels == clz)
            
            if mask.sum() > 0:
                boxes_clz = boxes[mask]
                scores_clz = scores[mask]
                labels_clz = labels[mask]
                
                boxes_nms, scores_nms, labels_nms = self.nms(boxes_clz, scores_clz, labels_clz)
                
                boxes_list.extend(boxes_nms)
                scores_list.extend(scores_nms)
                labels_list.extend(labels_nms)

        result = pd.DataFrame({
            'name': df['name'].iloc[0],
            'x_center': [box[0] for box in boxes_list],
            'y_center': [box[1] for box in boxes_list],
            'width': [box[2] for box in boxes_list],
            'height': [box[3] for box in boxes_list],
            'score': scores_list,
            'label': labels_list
        })
        
        return result

class Co_DETR_Sahi:
    def __init__(self, 
                 images_dir: str=None,
                 output_dir: str=None,
                 config_path: str='./configs/Co-DETR/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py',
                 weights_path: str='./weights/Co-DETR/co_dino_5scale_lsj_swin_large_3x_coco.pth',
                 conf_thresh: float=0.1,
                 image_size: int=1920,
                 device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 no_sliced_prediction=False,
                 data_type="coco"):
        
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.weights_path = weights_path
        self.conf_thresh = conf_thresh
        self.image_size = image_size
        self.device = device
        self.model_type = 'mmdet'
        self.no_sliced_prediction = no_sliced_prediction
        if data_type not in ['coco', 'cityscapes']:
            print(f"Invalid data type. Use 'coco' or 'cityscapes'")
        else:
            self.data_type = data_type
        
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=self.model_type,
            model_path=self.weights_path,
            config_path=self.config_path,
            confidence_threshold=self.conf_thresh,
            image_size=self.image_size,
            device=self.device
        )
        
    def slice_infer(self, 
                    slice_height: int=1280, 
                    slice_width: int=1280, 
                    overlap_height_ratio: float=0.1, 
                    overlap_width_ratio: float=0.1):
        self.raw_det_dir = f"{self.output_dir}/backlog/raw_detections"
        os.makedirs(self.output_dir, exist_ok=True)
        predict(
            model_type='mmdet',
            model_path=self.weights_path,
            model_config_path=self.config_path,
            model_device=device,
            model_confidence_threshold=0.1,
            source=self.images_dir,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            novisual=True, 
            export_pickle=True,
            project=self.output_dir,
            name=self.raw_det_dir,
            no_sliced_prediction=self.no_sliced_prediction 
        )
        
    def class_label(self, label):
        if self.data_type == "cityscapes":
            mapping = {
                0: 'person', 
                1: 'rider', 
                2: 'car',   
                3: 'truck',   
                4: 'bus',  
                5: 'on_rails',   
                6: 'motorcycle',   
                7: 'bicycle'    
            }
            new_label = mapping.get(label, -1)
        elif self.data_type == "coco":
            mapping = {
                0: 'person',  
                1: 'bicycle',  
                2: 'car',  
                3: 'motorcycle', 
                5: 'bus',  
                7: 'truck'  
            }
            new_label = mapping.get(label, -1)
        return new_label
    
    def remap_category(self, label):
        if self.data_type == "cityscapes":
            mapping = {
                'person': 0, 
                'car': 2, 
                'bicycle': 1,   
                'rider': 4,   
                'truck': 2,  
                'bus': 2,   
                'motorcycle': 1
                }
            new_label = mapping.get(label, -1)
        elif self.data_type == "coco":
            mapping = {
                'person': 0,  
                'bicycle': 1,  
                'car': 2,  
                'motorcycle': 1, 
                'bus': 2,  
                'truck': 2  
            }
            new_label = mapping.get(label, -1)
        return new_label
    

        
    def parse_pickles_to_dataframe(self):
        data = []
        for path in glob.iglob(f"{self.raw_det_dir}/pickles/*.pickle"):
            with open(path, 'rb') as file:
                detections = pickle.load(file)
                for detection in detections:
                    bbox_str = str(detection.bbox)
                    coords_str = bbox_str[bbox_str.index('<')+1:bbox_str.index('>')].split(',')[0:4]
                    x1, y1, x2, y2 = map(float, [coord.strip("()") for coord in coords_str])
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    label = detection.category.id
                    if self.data_type == "coco":
                        if label in [0, 1, 2, 3, 5, 7]:
                            label = self.class_label(label)
                            score = detection.score.value*100
                            filename = os.path.basename(path)
                            data.append((filename.replace('.pickle', '.png'), x_center, y_center, width, height, label, score))
                    elif self.data_type == "cityscapes":
                        if label in [0, 1, 2, 3, 4, 6, 7]:
                            label = self.class_label(label)
                            score = detection.score.value*100
                            filename = os.path.basename(path)
                            data.append((filename.replace('.pickle', '.png'), x_center, y_center, width, height, label, score)) 
                
        self.df = pd.DataFrame(data, columns=['name', 'x_center', 'y_center', 'width', 'height', 'label', 'score'])
        self.df.to_csv(f"{self.output_dir}/backlog/raw_detections.tsv", sep='\t',index=False)
        
    def postprocess(self, nms_iou):
        self.detections = self.df.copy()
        self.detections['label'] = self.detections['label'].apply(lambda label: self.remap_category(label)) 
        self.detections.to_csv(f"{self.output_dir}/backlog/raw_detections_categories.tsv", sep='\t',index=False)

        nms_processor = NMSProcessor(iou_thresh=nms_iou)
        self.detections = self.detections.groupby('name').apply(nms_processor.nms_by_category).reset_index(drop=True)
        self.detections.to_csv(f"{self.output_dir}/detections.tsv", sep='\t',index=False)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Inference")
    
    parser.add_argument("--config_path", type=str, required=True, 
                        help="Path to the config file")
    parser.add_argument("--images_dir", type=str, required=True, 
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory")
    
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config_path)
    
    co_detr_sahi = Co_DETR_Sahi(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        config_path=config['MODEL']['model_config'],
        weights_path=config['MODEL']['model_weights'],
        conf_thresh=float(config['PARAMS']['conf_thresh']),
        image_size=int(config['PARAMS']['image_size']),
        no_sliced_prediction=config.getboolean('PARAMS', 'no_sliced_prediction'),
        data_type=config['PARAMS']['data_type']
    )
    
    co_detr_sahi.slice_infer(
        slice_height=int(config['PARAMS']['slice_height']),
        slice_width=int(config['PARAMS']['slice_width']),
        overlap_height_ratio=float(config['PARAMS']['overlap_height_ratio']),
        overlap_width_ratio=float(config['PARAMS']['overlap_width_ratio'])
    )
    
    co_detr_sahi.parse_pickles_to_dataframe()
    co_detr_sahi.postprocess(config['PARAMS']['nms_iou'])