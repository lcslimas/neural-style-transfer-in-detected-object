from ultralytics import YOLO

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = YOLO('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='data.yaml', epochs=10000, imgsz=640)