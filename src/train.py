import os
import pandas as pd
import numpy as np

import albumentations
import torch
import pytorch_pfn_extras as ppe

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold

import dataset
import engine
from Nets import get_model
import config


if __name__ == "__main__":
    
# # # make a list of train dataset   list[path,cry]
    tmp_list = []

    for label in os.listdir(config.TRAIN):
        path = config.TRAIN + label + '/'
        for wavefile in os.listdir(path):
            path_tmp=''
            path_tmp = path + wavefile
            tmp_list.append([path_tmp,label])
    train_file = pd.DataFrame(tmp_list,columns=['file_path','label'])
    #print(train_file)
    del tmp_list
    
# # # split
    skf = StratifiedKFold(**config.split)
    train_file['fold']=-1
    for fold_id, (tr_ind, val_ind) in enumerate(skf.split(train_file, train_file['label'])):
        train_file.iloc[val_ind,-1] = fold_id
    #print(train_file['fold'])
    use_fold = config.globals["use_fold"]
    train_file_list = train_file.query("fold != @use_fold")[["file_path", "label"]].values.tolist()
    val_file_list = train_file.query("fold == @use_fold")[["file_path", "label"]].values.tolist()

    #print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file_list), len(val_file_list)))

    engine.set_seed(config.globals["seed"])
    device = torch.device(config.globals["device"])
    
# # # get loader
    train_loader, val_loader = dataset.get_loaders_for_training(
        config.dataset["params"], config.loader, train_file_list, val_file_list)

# # # get model
    model = get_model(config.model)
    model = model.to(device)

# # # get optimizer
    optimizer = getattr(
        torch.optim, 
        config.optimizer["name"]
        )(model.parameters(), **config.optimizer["params"])

# # # get scheduler
    scheduler = getattr(
        torch.optim.lr_scheduler, 
        config.scheduler["name"]
        )(optimizer, **config.scheduler["params"])

# # # get loss
    loss_func = getattr(torch.nn, config.loss["name"])(**config.loss["params"])

# # # create training manager
    trigger = None

    manager = ppe.training.ExtensionsManager(
        model, optimizer, config.globals["num_epochs"],
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger,
        out_dir=config.MODEL_OUT
    )
    settings={}
# # # set manager extensions
    manager = engine.set_extensions(
        manager, settings, model, device,
        val_loader, optimizer, loss_func,
    )

# # # training
    engine.train(
        manager, 
        settings,
        train_loader, 
        model, 
        optimizer,
        device, 
        scheduler, 
        loss_func,
    )
    