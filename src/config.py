MODEL_OUT = '../models/'
TRAIN = '../input/train/'
TEST = '../input/test/'
SUBMISSION = '../sub/'


globals = {
  'seed' : 1213,
  'device' : 'cuda',
  'num_epochs' : 50,
  'use_fold' : 0,
  'target_sr' : 32000,
}
#main_metric: epoch_f1
#minimize_metric: False
#input_key: image
#input_target_key: targets
#weights:
#folds:



split = {
  'n_splits':5,
  'random_state':42,
  'shuffle':True,
}



model = {
    'name': 'resnest50_fast_1s1x64d',
    'params':{
        'pretrained':True,
        'n_classes':6,
    }
}

dataset={
  'params':{
    'img_size' : 224,
    'melspectrogram_params':{
      'n_mels' : 128,
      'fmin' : 20,
      'fmax' : 16000,
      }
  }
}

loader={
  'train':{
    'batch_size':8,
    'shuffle': True,
    'pin_memory': True,
    'num_workers':4
    },

  'valid':{
    'batch_size' : 16,
    'shuffle' : False,
    'pin_memory' : True,
    'num_workers' : 4,
  }
}

optimizer= {
  'name': 'Adam', 
  'params':{
    'lr' : 0.001,
  }
}

scheduler = {
  'name':'CosineAnnealingLR',
  'params':{
    'T_max' : 10,
    }
}

loss = {
  'name': 'BCEWithLogitsLoss', # 'ResNetLoss'
  'params':{}
}
#loss_type = 'bce'
#loss_type = 'BCEWithLogitsLoss'


'''
callbacks:
  - name: F1Callback
    params:
      input_key: targets
      output_key: logits
      model_output_key: multilabel_proba
      prefix: f1
  - name: mAPCallback
    params:
      input_key: targets
      output_key: logits
      model_output_key: multilabel_proba
      prefix: mAP
'''