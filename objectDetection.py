from __future__ import print_function

from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor
import numpy as np
import fastNeuralStyleTransfer as fnst


# Load a model
model = YOLO("yolov8x-seg.pt")

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
      
      # stylizedImg = stylizedImg.astype(np.uint8)
      cv2.imshow('Imagem original', ori_img)
      stylizedImg[allMasks == 0] = 0
      ori_img[allMasks != 0] = 0
      stylizedImg = np.clip(stylizedImg * 255.0, 0, 255).astype(np.uint8)
      blended = cv2.addWeighted(ori_img, 1, stylizedImg, 1, 0)
      cv2.imshow('Imagem com transferência Neural de estilo', stylizedImg)
      cv2.imshow('Imagem original com tne', blended)
    
# results = model.predict(source= "0", show= True, boxes= False)
model.add_callback("on_predict_batch_start", on_predict_batch_start)

model.predict(source="0", boxes= False, show= False, classes = 0)
