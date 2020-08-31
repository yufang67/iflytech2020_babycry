import os
import torch

import numpy as np
import torch
import torch.nn as nn
import resnest.torch as resnest_torch
import torch.utils.data as data
from torch.nn import functional as F
import cv2

import src.dataset as dataset
from src.dataset import SpectrogramDataset
import src.config as config

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER = "/home/yufang/app/static/"
#UPLOAD_FOLDER = "/home/yfang/iflytech2020_babycry/static/"
DEVICE = 'cpu'

# # # Model
def resnest_model():

    model = getattr(resnest_torch, 'resnest50_fast_1s1x64d')(pretrained=None)
    del model.fc
    model.fc = nn.Sequential(
    nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Linear(1024, 6), nn.Softmax(),
    )
    return model

# # # Prediction
def predict(wave_path):

    test_wave = [[wave_path,'awake']] # random label
    test_dataset = SpectrogramDataset(test_wave, **config.dataset["params"])
    test_loader = data.DataLoader(test_dataset, **config.loader["test"])

    model = resnest_model()
    state_dict = torch.load('models/snapshot_epoch_46.pth',map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader: # random label is not considered
            inputs = inputs.to(DEVICE)
            prediction =  model(inputs)
        
    return prediction.numpy().ravel(), inputs.numpy()

# # # web app
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        wave_file = request.files["wave_file"]
        if wave_file:
            wave_location = os.path.join(
                UPLOAD_FOLDER,
                wave_file.filename
            )
            wave_file.save(wave_location)
            pred_ohprob, image = predict(wave_location)
            
            img_location = os.path.join(UPLOAD_FOLDER,'img.jpg')
            image = image.squeeze(axis=0)
            image = np.moveaxis(image,0,-1)
            image = image*225
            cv2.imwrite(img_location, image)
            
            
            pred_dict = {}
            for i in range(len(pred_ohprob)):
                pred_dict[dataset.INV_CRY_CODE[i]]=pred_ohprob[i]

            #print(wave_file.filename)
            return render_template(
                "index.html", 
                prediction=pred_ohprob, 
                wave_loc=wave_file.filename,
                pred_dict=pred_dict,
                img_loc='img.jpg'
                )
    return render_template("index.html", prediction=0, wave_loc=None, pred_dict=None)

if __name__ == "__main__":

    app.run(host='0.0.0.0',port=15000,debug=True)
    #app.run(debug=True)








