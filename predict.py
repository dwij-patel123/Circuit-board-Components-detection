import cv2
import torch,torchvision
import detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt


loaded_cfg = get_cfg()
loaded_cfg = loaded_cfg.load_yaml_with_base('content/output/config.yaml')
loaded_cfg = detectron2.config.CfgNode(loaded_cfg)

print(torchvision.__version__)

loaded_cfg["MODEL"]["WEIGHTS"] = 'content/output/model_final.pth'
loaded_cfg["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"] = 0.1
loaded_cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(loaded_cfg)
img = cv2.imread('upload/a-circuit-board-description-automatically-generat.jpeg')
print(img.shape)
outputs = predictor(img)
visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
# check for detectron2 docs how to get image from visualizer VisImage.get_image()
cv2.imshow(winname="pred_img",mat=out.get_image())
cv2.waitKey(0)



