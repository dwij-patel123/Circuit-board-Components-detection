from flask import Flask, flash, request, redirect, url_for,render_template
import os
import cv2
import torch,torchvision
import detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


UPLOAD_FOLDER = 'upload'
# essential cause flask said so to put into folder named static otherwise it will not render
PREDICTED_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



app = Flask(__name__,template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICTED_FOLDER

def predict(file_name):
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'],file_name))
    #Infering on the give image filename in parameters
    loaded_cfg = get_cfg()
    loaded_cfg = loaded_cfg.load_yaml_with_base('content/output/config.yaml')
    loaded_cfg = detectron2.config.CfgNode(loaded_cfg)
    loaded_cfg["MODEL"]["WEIGHTS"] = 'content/output/model_final.pth'
    loaded_cfg["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"] = 0.2
    loaded_cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(loaded_cfg)
    outputs = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    # defining path to write image
    pred_img_path = os.path.join(app.config['PREDICTED_FOLDER'],"pred_"+file_name)
    cv2.imwrite(filename=pred_img_path,img=out.get_image())
    return pred_img_path



@app.route("/hello")
def hello_world():
    return "<p>hello world</p>"

# to render index.html
@app.route("/",methods=['GET'])
def home():
    return render_template("index.html")

# to upload file
@app.route("/upload_img",methods = ['POST'])
def upload_file_and_predict():
    if request.method == 'POST':
        f = request.files['img_file']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        f.save(img_path)
        pred_img_path = predict(f.filename)
        return render_template("result.html",pred_img = pred_img_path)

