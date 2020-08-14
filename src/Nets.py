
import torch.nn as nn
import torch.nn.functional as F
import torch
import resnest.torch as resnest_torch

from torchvision import models


class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=6):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = F.sigmoid(x)
        return {
            "logits": x,
            "multiclass_proba": multiclass_proba,
            "multilabel_proba": multilabel_proba
        }


def get_model(model_config: dict):

    model_name = model_config["name"]
    model_params = model_config["params"]

    if model_name=="resnet":
        model = ResNet(  # type: ignore
            base_model_name=model_name,
            pretrained=model_params["pretrained"],
            num_classes=model_params["n_classes"])
        return model

    elif model_name=='resnest50_fast_1s1x64d':
        model = getattr(resnest_torch, model_name)(pretrained=model_params["pretrained"])
        del model.fc
        # # use the same head as the baseline notebook.
        model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, model_params["n_classes"]))
        return model
    else:
        raise NotImplementedError
        

