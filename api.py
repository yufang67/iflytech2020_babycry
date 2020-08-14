import os
import torch

import albumentations
import numpy as pd
import torch.nn as nn
from torch.nn import functional as F

#import dataset
#import engine

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER = "home/yufang/Example_babycry/static"
DEVICE = 'cpu'
MODEL = None

# # # Model
class ResNest(nn.Module):
    def __init__(self,pretrained=None):
        pass

# # # Prediction
def predict(wave_path,model):
    pass

# # # web app
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        wave_file = request.files["images"]
        if wave_file:
            wave_location = os.path.join(
                UPLOAD_FOLDER,
                wave_file.filename
            )
            wave_file.save(wave_location)
            pred = predict(wave_location, MODEL)[0]
            return render_template("index.html", prediction=pred, wave_loc=wave_file.filename)
    return render_template("index.html", prediction=0, wave_loc=None)

if __name__ == "__main__":
    MODEL = ResNest()
    #MODEL.load_state_dict(torch.load("model.bin",map_location=torch.device(Device)))
    MODEL.to(DEVICE)

    app.run(host='0.0.0.0',port=12000,debug=True)








