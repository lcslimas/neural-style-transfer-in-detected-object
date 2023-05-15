from __future__ import print_function

from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor
import numpy as np
import fastNeuralStyleTransfer as fnst


# Load a model
model = YOLO("yolov8s-seg.pt")

def on_predict_batch_start(predictor: SegmentationPredictor):
  for result in (predictor.results if hasattr(predictor, "results") else []):
    if(result.masks is not None):
      mask = result.masks.cpu().numpy()
      masks = mask.data.astype(bool)
      ori_img = result.orig_img
      allMasks = ""
      for m in masks:
        new = np.zeros_like(ori_img, dtype=np.uint8)
        
        new[m] = ori_img[m]
        if(allMasks == "") :
          allMasks = new
        else :
          allMasks = cv2.addWeighted(allMasks, 1, new, 1, 0)
      
      stylizedImg = np.transpose(fnst.test_image(allMasks,
           checkpoint_model = './checkpoints/best_model.pth',
           save_path = './')[0], (1,2,0))[...,::-1]
      # blended = cv2.addWeighted(ori_img, 1, stylizedImg, 0.5, 0)
      # cv2.imshow('shape + img', blended)
      cv2.imshow('shape + img', stylizedImg)
      cv2.imshow('img', ori_img)
    
# results = model.predict(source= "0", show= True, boxes= False)
model.add_callback("on_predict_batch_start", on_predict_batch_start)

model.predict(source="0", boxes= False, show= False, classes = 0)
