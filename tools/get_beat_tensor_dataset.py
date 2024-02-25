import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import glob
import os
import cv2


def get_beat_tensor_dataset_pad(anno_path: str, load_folder: str, sv_pth: str):
    annos = pd.read_csv(anno_path)
    x_list = list()
    y_list = list()
    l_list = list()
    for i in range(len(annos)):
        id = str(int(annos["musicId"][i]))
        file_path = os.path.join(load_folder, id+".npy")

        x = np.load(file_path)
        y = np.array([annos["Arousal(mean)"][i], annos["Valence(mean)"][i]])
        l_list.append(len(x))
        x = torch.from_numpy(x)
        x = x.float()
        x_list.append(x)
        y_list.append(y)

    x = pad_sequence(x_list, batch_first=True)
    l = torch.tensor(l_list)
    y = np.array(y_list)
    y = torch.from_numpy(y)
    y = y.float()

    dic = dict()
    dic["x"] = x
    dic["y"] = y
    dic["l"] = l
    torch.save(dic, sv_pth)


def get_beat_tensor_dataset_rsz(anno_path: str, load_folder: str, sv_pth: str, to_len: int = 70):
    annos = pd.read_csv(anno_path)
    x_list = list()
    y_list = list()
    for i in range(len(annos)):
        id = str(int(annos["musicId"][i]))
        file_path = os.path.join(load_folder, id+".npy")

        x = np.load(file_path)
        x = cv2.resize(x, (1, to_len))
        x = np.reshape(x, (to_len,))
        y = np.array([annos["Arousal(mean)"][i], annos["Valence(mean)"][i]])
        x = torch.from_numpy(x)
        x = x.float()
        x_list.append(x)
        y_list.append(y)

    x = pad_sequence(x_list, batch_first=True)
    y = np.array(y_list)
    y = torch.from_numpy(y)
    y = y.float()

    dic = dict()
    dic["x"] = x
    dic["y"] = y
    dic["l"] = to_len
    torch.save(dic, sv_pth)


if __name__ == "__main__":
    to_len = 75
    annos_path = "Data/PMEmo2019/annotations/static_annotations.csv"
    from_pth = "Data/Beat_dataset/Beat_after_PMEmo_no_head"
    to_pth = "Data/Beat_dataset/Beat_tensor_dataset/beat_no_head_rsz_" + \
        str(to_len)+".pt"
    get_beat_tensor_dataset_rsz(annos_path, from_pth, to_pth, to_len)
