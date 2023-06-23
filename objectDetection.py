from __future__ import print_function

from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor
import numpy as np
import fastNeuralStyleTransfer as fnst


# Load a model
model = YOLO("yolov8x-seg.pt")

styles = ['Romero_Britto_5600.pth', 'Picasso_Selfportrait_5000.pth' ]

def on_predict_batch_start(predictor: SegmentationPredictor):
  for result in (predictor.results if hasattr(predictor, "results") else []):
    if(result.masks is not None):
      mask = result.masks.cpu().numpy()
      masks = mask.data.astype(bool)
      ori_img = result.orig_img
      allMasks = ""
      allStyledMasks = ""
      index = 0
      for m in masks:
        new = np.zeros_like(ori_img, dtype=np.uint8)
        new[m] = ori_img[m]
        stylizedImg = np.transpose(fnst.test_image(new,
           checkpoint_model = './checkpoints/' + styles[0 if index % 2 == 0 else 1]  ,
           save_path = './')[0], (1,2,0))[...,::-1]
        stylizedImg = np.clip(stylizedImg * 255.0, 0, 255).astype(np.uint8)
        stylizedImg[m == False] = 0
        if(allMasks == "" and allStyledMasks == "") :
          allMasks = new
          allStyledMasks = stylizedImg
        else :
          allMasks = cv2.addWeighted(allMasks, 1, new, 1, 0)
          allStyledMasks = cv2.addWeighted(allStyledMasks, 1, stylizedImg, 1, 0)
        index +=1
      cv2.imshow('Imagem original', ori_img)
      ori_img[allMasks != 0] = 0
      blended = cv2.addWeighted(ori_img, 1, allStyledMasks, 1, 0)
      cv2.imshow('Imagem original com tne', blended)
    
model.add_callback("on_predict_batch_start", on_predict_batch_start)

model.predict(source="0", boxes= False, show= False, classes = 0)
