from __future__ import print_function

from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor
import numpy as np
import fastNeuralStyleTransfer as fnst
from time import perf_counter
# Load a model
model = YOLO("yolov8x-seg.pt")

styles = ['romero_britto_improved_6000.pth', 'Picasso_Selfportrait_6000_improved.pth' ]

def on_predict_batch_start(predictor: SegmentationPredictor):
  allStyledMask0 = None
  allStyledMask1 = None
  for result in (predictor.results if hasattr(predictor, "results") else []):
    start = perf_counter()
    if(result.masks is not None):
      mask = result.masks.cpu().numpy()
      masks = mask.data.astype(bool)
      ori_img = result.orig_img
      index = 0
      maskShape = masks[0].shape
      ori_img = cv2.resize(ori_img, (maskShape[1], maskShape[0]))
      ori_img = ori_img.reshape((maskShape[0], maskShape[1], 3))
      objectsArr = np.array(result.boxes.cls.cpu())
      for m in masks:
        new = np.zeros_like(ori_img, dtype=np.uint8)
        new[m] = ori_img[m]
        style_index = 0 if objectsArr[index] % 2 == 0 else 1
        if(allStyledMask0 is None and style_index == 0):
            allStyledMask0 = new
        elif(allStyledMask1 is None and style_index == 1) : 
            allStyledMask1 = new
        else :
          if(style_index == 0): 
            allStyledMask0 = cv2.addWeighted(allStyledMask0, 1, new, 1, 0)
          else: 
            allStyledMask1 = cv2.addWeighted(allStyledMask1, 1, new, 1, 0)
        index +=1

      allStyledMask0Tne = None
      allStyledMask1Tne = None
      cv2.imshow('Imagem original', ori_img)
      if(allStyledMask0 is not None):
        allStyledMask0Tne = np.transpose(fnst.test_image(allStyledMask0,
          checkpoint_model = './checkpoints/' + styles[0]  ,
          save_path = './')[0], (1,2,0))[...,::-1]
        allStyledMask0Tne = np.clip(allStyledMask0Tne * 255.0, 0, 255).astype(np.uint8)
        allStyledMask0Tne[allStyledMask0 == 0] = 0
        ori_img[allStyledMask0 != 0] = 0
      
      if(allStyledMask1 is not None):
        allStyledMask1Tne = np.transpose(fnst.test_image(allStyledMask1,
          checkpoint_model = './checkpoints/' + styles[1]  ,
          save_path = './')[0], (1,2,0))[...,::-1]
        allStyledMask1Tne = np.clip(allStyledMask1Tne * 255.0, 0, 255).astype(np.uint8)
        allStyledMask1Tne[allStyledMask1 == 0] = 0
        ori_img[allStyledMask1 != 0] = 0
      
      if(allStyledMask0Tne is not None and allStyledMask1Tne is not None):
        stylizedImg = cv2.addWeighted(allStyledMask0Tne, 1, allStyledMask1Tne, 1, 0)
      else:
        stylizedImg = allStyledMask0Tne if allStyledMask0Tne is not None else allStyledMask1

      blended = cv2.addWeighted(ori_img, 1, stylizedImg, 0.7, 0)
      cv2.imshow('Imagem original com tne', blended)
      cv2.waitKey(1)
      print("velocidade de execução: " + str(1 / (perf_counter() - start) ))
      
    
model.add_callback("on_predict_batch_start", on_predict_batch_start)

model.predict(source="C:/Users/lcsli/Downloads/190111_04_TaksinBridge_HD_03.mp4", boxes= False, show= False, stream= False)
