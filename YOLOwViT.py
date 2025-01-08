import kagglehub
andrewmvd_face_mask_detection_path = kagglehub.dataset_download('andrewmvd/face-mask-detection')

print('Data source import complete.')
import os
import torch
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder

!pip install ultralytics transformers


import wandb
wandb.login(key='c6173cf27280b68511e3c01d809d179681002c51')

ANNOTATIONS_DIR = "/root/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1/annotations"
IMAGES_DIR = "/root/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1/images"
OUTPUT_DIR = "/kaggle/working/face-mask-detection-yolo"

os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels', 'val'), exist_ok=True)

annotations = []
for xml_file in os.listdir(ANNOTATIONS_DIR):
    tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
    root = tree.getroot()
    file_name = root.find('filename').text
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((class_name, xmin, ymin, xmax, ymax))
    annotations.append((file_name, objects))

train_annotations = annotations[:int(0.8*len(annotations))]
val_annotations = annotations[int(0.8*len(annotations)):]

label_encoder = LabelEncoder()
all_labels = [obj[0] for ann in annotations for obj in ann[1]]
label_encoder.fit(all_labels)

def convert_to_yolo_format(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

def save_yolo_files(annotations, image_dir, label_dir):
    for file_name, objects in annotations:
        img_path = os.path.join(IMAGES_DIR, file_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        shutil.copy(img_path, image_dir)
        label_path = os.path.join(label_dir, file_name.replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            for obj in objects:
                class_name, xmin, ymin, xmax, ymax = obj
                class_id = label_encoder.transform([class_name])[0]
                bb = convert_to_yolo_format((w, h), (xmin, ymin, xmax, ymax))
                f.write(f"{class_id} {' '.join(map(str, bb))}\n")

save_yolo_files(train_annotations, os.path.join(OUTPUT_DIR, 'images', 'train'), os.path.join(OUTPUT_DIR, 'labels', 'train'))
save_yolo_files(val_annotations, os.path.join(OUTPUT_DIR, 'images', 'val'), os.path.join(OUTPUT_DIR, 'labels', 'val'))

data_yaml = f"""
train: {os.path.join(OUTPUT_DIR, 'images', 'train')}
val: {os.path.join(OUTPUT_DIR, 'images', 'val')}

nc: {len(label_encoder.classes_)}
names: {label_encoder.classes_.tolist()}
"""

with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
    f.write(data_yaml)

from transformers import ViTModel
import torch.nn as nn

class ViTBackbone(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super(ViTBackbone, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.features = self.vit.embeddings
        self.transformer = self.vit.encoder.layer

    def forward(self, x):
        x = self.features(x)
        for layer in self.transformer:
            x = layer(x)[0]
        return x

from ultralytics.yolo.models.common import Detect, BaseModel
from ultralytics.yolo.utils.torch_utils import initialize_weights

class YOLOv8WithViT(BaseModel):
    def __init__(self, num_classes, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.backbone = ViTBackbone(model_name)
        self.detect = Detect(num_classes=num_classes, ch=[768])
        initialize_weights(self.detect)

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detect(features)
        return detections

from ultralytics import YOLO

custom_model = YOLOv8WithViT(num_classes=len(label_encoder.classes_))
pretrained_yolo = YOLO('yolov8n.pt')
custom_model.detect.load_state_dict(pretrained_yolo.model.detect.state_dict())

custom_model.train(data=os.path.join(OUTPUT_DIR, 'data.yaml'),
                   epochs=80,
                   imgsz=416,
                   batch=16,
                   cache=True,
                   device=0)

