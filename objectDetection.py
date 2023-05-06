from __future__ import print_function

from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor
import numpy as np
import torch


# Load a model
model = YOLO("yolov8s-seg.pt")

def on_predict_batch_start(predictor: SegmentationPredictor):
  for result in (predictor.results if hasattr(predictor, "results") else []):
    if(result.masks is not None):
      mask = result.masks.cpu().numpy()
      masks = mask.masks.astype(bool)
      ori_img = result.orig_img
      allMasks = ""
      for m in masks:
        new = np.zeros_like(ori_img, dtype=np.uint8)
        
        new[m] = ori_img[m]
        if(allMasks == "") :
          allMasks = new
        else :
          allMasks = cv2.addWeighted(allMasks, 1, new, 1, 0)
          

      blended = cv2.addWeighted(ori_img, 1, allMasks, -0.5, 0)
      cv2.imshow('shape + img', blended)
    
# results = model.predict(source= "0", show= True, boxes= False)
model.add_callback("on_predict_batch_start", on_predict_batch_start)

model.predict(source="0", boxes= False, show= False)
