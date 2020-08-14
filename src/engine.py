import torch
import torch.utils.data as data

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions

from tqdm import tqdm
from contextlib import contextmanager
import os
import random
import time
import numpy as np


def train(
    manager, args, 
    data_loader, model, 
    optimizer, device,
    scheduler, loss_func
    ):

    """Run minibatch training loop"""
    while not manager.stop_trigger:
        model.train()
        for batch_idx, (inputs,target) in enumerate(data_loader):
            with manager.run_iteration():
                #inputs = data['images']
                #targets = data['targets']

                inputs, target = inputs.to(device,dtype=torch.float), target.to(device,dtype=torch.float)
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_func(output, target)
                ppe.reporting.report({'train/loss': loss.item()})
                loss.backward()
                optimizer.step()
                scheduler.step()

def evaluate(
    args, 
    model, 
    device,
    inputs,
    target,
    loss_func, 
    eval_func_dict={}
):
    """
    Run evaliation for valid
    
    This function is applied to each batch of val loader.
    """
    final_targets = []
    final_outputs = []
    correct = 0

    model.eval()
    #with torch.no_grad():

        #for data in data_loader:
    #inputs = data['images']
    #targets = data['targets']

    inputs, target = inputs.to(device,dtype=torch.float), target.to(device,dtype=torch.float)
           
    output = model(inputs)
# Final result will be average of averages of the same size
    val_loss = loss_func(output, target).item()
    ppe.reporting.report({'val/loss': val_loss})

    pred = output.argmax(dim=1)
    correct += pred.eq(target.argmax(dim=1)).sum().item()
    #print(inputs.shape)
    ppe.reporting.report({'val/acc': correct/inputs.shape[0]})
    #for eval_name, eval_func in eval_func_dict.items():
    #    eval_value = eval_func(output, targets).item()
    #    ppe.reporting.report({"val/{}".format(eval_name): eval_value})

            #targets = targets.detach().cpu().numpy().tolist()
            #output = output.detach().cpu().numpy().tolist()
                
            #final_targets = final_targets.extend(targets)
            #final_outputs = final_outputs.extend(output)
    #return final_outputs, final_targets


def set_extensions(
    manager, args, model, device, test_loader, optimizer,
    loss_func, eval_func_dict={}
):
    """set extensions for PPE"""
        
    my_extensions = [
        # # observe, report
        ppe_extensions.observe_lr(optimizer=optimizer),
        # ppe_extensions.ParameterStatistics(model, prefix='model'),
        # ppe_extensions.VariableStatisticsPlot(model),
        ppe_extensions.LogReport(),
        ppe_extensions.PlotReport(['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        ppe_extensions.PlotReport(['lr',], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            'epoch', 'iteration', 'lr', 'train/loss', 'val/loss', 'val/acc', "elapsed_time"]),
        #ppe_extensions.ProgressBar(update_interval=100),

        # # evaluation
        (
            ppe_extensions.Evaluator(
                test_loader, model,
                eval_func=lambda inputs, target : evaluate(args, model, device,inputs,target, loss_func, eval_func_dict),
                progress_bar=True),
                (1, "epoch"),
                ),
        # # save model snapshot.
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_{.updater.epoch}.pth"),
                ppe.training.triggers.MinValueTrigger(key="val/loss", trigger=(1, 'epoch'))
        ),
    ]
           
    # # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)
        
    return manager

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
#     torch.backends.cudnn.deterministic = True  # type: ignore
#     torch.backends.cudnn.benchmark = True  # type: ignore
    

@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))