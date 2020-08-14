import cv2
import librosa
import numpy as np
#import pandas as pd
import soundfile as sf
import torch.utils.data as data

from pathlib import Path
import typing as tp
import torch


CRY_CODE = {
    'awake':0,
    'diaper':1,
    'hug':2,
    'hungry':3,
    'sleepy':4,
    'uncomfortable':5,
}

INV_CRY_CODE = {v:k for k, v in CRY_CODE.items()}

PERIOD = 5

class SpectrogramDataset(data.Dataset):
    def __init__(
        self,
        file_list:tp.List[tp.List[str]],
        img_size=224,
        waveform_transforms=None,
        spectrogram_transforms=None,
        melspectrogram_params={}
        ):
        self.file_list = file_list # list of list: [file_path, cry_code]
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_params = melspectrogram_params

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx:int):
        wave_path, cry_code = self.file_list[idx]
        y, _ = sf.read(wave_path)
        sr = 44100 #16000
        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            eff_len = sr * PERIOD
            if len_y < eff_len:
                new_y = np.zeros(eff_len,dtype=y.dtype)
                start = np.random.randint(eff_len - len_y)
                new_y[start:start+len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > eff_len:
                start = np.random.randint(len_y-eff_len)
                y = y[start:start+eff_len].astype(np.float32)
            else:
                y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y,sr=sr,**self.melspectrogram_params)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        img = mono_to_color(melspec)
        h, w, _ = img.shape # keep ratio = h/w
        
        img = cv2.resize(img, (int(w*self.img_size/h),self.img_size))
        #print(img.shape,sr)
        
        img = np.moveaxis(img,2,0) # (a, source, destination)
        img = (img/255.0).astype(np.float32)
        labels = np.zeros(len(CRY_CODE),dtype="f")
        labels[CRY_CODE[cry_code]] = 1

        return img, labels


def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def get_loaders_for_training(
    args_dataset: tp.Dict, args_loader: tp.Dict,
    train_file_list: tp.List[str], val_file_list: tp.List[str]
):
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, **args_dataset)
    # # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    val_loader = data.DataLoader(val_dataset, **args_loader["valid"])
    
    return train_loader, val_loader
