from ultralytics import YOLO
from ultralytics.yolo.v8.segment.predict import SegmentationPredictor
import torchvision.transforms as T

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load a model
model = YOLO("yolov8n-seg.pt")

def show_mask(mask):
  transform = T.ToPILImage()
  # img = transform(mask)

def on_predict_batch_start(predictor: SegmentationPredictor):
  # print(predictor.preprocess())
  if(hasattr(predictor, "results")):
    print(predictor)
    # predictor.results[-1].masks.masks = predictor.results[0].masks.masks
    # show_mask(predictor.results[-1].masks.masks)
    


# results = model.predict(source= "0", show= True, boxes= False)
model.add_callback("on_predict_batch_start", on_predict_batch_start)

model.predict(source="0", boxes= False, show= True, retina_masks = True)


    


# print(results)