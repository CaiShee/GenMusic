import pandas as pd
import numpy as np
import pandasrw as pdrw
import os
import glob
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time


def train(dataloader: DataLoader, model: nn.Module,
          criterion: nn.modules.loss._Loss,
          optimizer: torch.optim.Optimizer, epoch: int,
          device: torch.device,
          autoSave: bool = True, autoPrint: int = 10, autoLog=True) -> nn.Module:
    """_summary_

    Returns:
        _type_: _description_
    """
    log = ""
    model = model.to(device)
    criterion = criterion.to(device)

    ticks = time.time()
    file_name = os.path.join("runs", str(ticks))
    if (not os.path.exists(file_name)):
        os.makedirs(file_name)
    if autoLog:
        file = open(os.path.join(file_name, "log.txt"), 'w')

    best_value = float("inf")
    for i in range(epoch):
        loss_value = 0
        for id, (feature, target) in enumerate(dataloader):
            feature = feature.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
        loss_value /= (id+1)

        src = "第 {} 轮误差: {}".format(i+1, loss_value)
        log += src+"\n"
        if autoPrint:
            if (i+1) % autoPrint == 0:
                print(6*"==== "+"\n"+src)
        if loss_value < best_value:
            best_value = loss_value
            if autoSave:
                torch.save(model, os.path.join(file_name, "best.pt"))
    if autoLog:
        file.write(log)
        file.close()
    if autoSave:
        torch.save(model, os.path.join(file_name, "last.pt"))
    return model


def pt2onnx(pt_path: str, *input_dims: int, device: torch.device = torch.device("cpu")):

    dir_name = os.path.dirname(pt_path)
    file_name = os.path.basename(pt_path)
    file_name = file_name.split(".")[0]

    new_name = os.path.join(dir_name, file_name+".onnx")

    model = torch.load(pt_path)
    model = model.to(device)
    model.eval()

    x = torch.randn(*input_dims)
    x = x.to(device)
    x = x.to(torch.float32)

    torch.onnx.export(model, x, new_name)


if __name__ == "__main__":
    ...
