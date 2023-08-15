import sys
import os
import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv_custom
import cv2
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from IPython.display import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
import os
import pickle
import pandas as pd
import argparse
import glob

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Co_DETR_Sahi:
    def __init__(self, 
                 images_dir: str=None,
                 output_dir: str=None,
                 config_path: str='./configs/Co-DETR/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py',
                 weights_path: str='./weights/Co-DETR/co_dino_5scale_lsj_swin_large_3x_coco.pth',
                 conf_thresh: float=0.1,
                 image_size: int=1920,
                 device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.weights_path = weights_path
        self.conf_thresh = conf_thresh
        self.image_size = image_size
        self.device = device
        self.model_type = 'mmdet'
        
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
        self.raw_det_dir = f"{self.output_dir}/backlog/Co-DETR/raw_detections"
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
        )
        
    def parse_pickles_to_dataframe(self):
        data = []
        for filename in glob.iglob(f"{self.raw_det_dir}/pickles/*.pickle"):
            with open(os.path.join(folder_path, filename), 'rb') as file:
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
                    if label in [0, 1, 2, 3, 5, 7]:
                        score = detection.score.value*100
                        data.append((filename.replace('.pickle', '.png'), x_center, y_center, width, height, label, score))
                        
        self.df = pd.DataFrame(data, columns=['name', 'x_center', 'y_center', 'width', 'height', 'label', 'score'])
        self.df.to_csv(f"{self.output_dir}/co_detr.tsv", sep='\t',index=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-DETR Sahi Object Detection")
    
    parser.add_argument("--images_dir", type=str, required=True, 
                        default='/home/ubuntu/workspace/datasets/ab_car_validation_germany/2w_misses', help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, 
                        default= '/home/ubuntu/workspace/results/ab_car_validation_germany/riders/Co-DETR/2w_misses_codetr', help="Output directory")
    parser.add_argument("--config_path", type=str, default='./configs/Co-DETR/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py', help="Path to config file")
    parser.add_argument("--weights_path", type=str, default='./weights/Co-DETR/co_dino_5scale_lsj_swin_large_3x_coco.pth', help="Path to weights file")
    parser.add_argument("--conf_thresh", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--image_size", type=int, default=3840, help="Image size")
    parser.add_argument("--slice_height", type=int, default=1280, help="Slice height")
    parser.add_argument("--slice_width", type=int, default=1280, help="Slice width")
    parser.add_argument("--overlap_height_ratio", type=float, default=0.1, help="Overlap height ratio")
    parser.add_argument("--overlap_width_ratio", type=float, default=0.1, help="Overlap width ratio")
    
    args = parser.parse_args()

    co_detr_sahi = Co_DETR_Sahi(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        weights_path=args.weights_path,
        conf_thresh=args.conf_thresh,
        image_size=args.image_size
    )
    
    co_detr_sahi.slice_infer(
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio
    )
    
    co_detr_sahi.parse_pickles_to_dataframe()