# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import os
import re
from PIL import Image
import numpy as np
import pandas as pd
from typing import TypeVar, Generic

tensor = TypeVar('tensor')


DEPTH_DIVIDING = 256.0
ACC_EPSILON = 1e-9


def read_label_list(list_path: str):
    input_dataframe = pd.read_csv(list_path)
    return input_dataframe["gt_disp"].values


def d_1(res: tensor, gt: tensor, start_threshold: int = 2,
        threshold_num: int = 4, relted_error: float = 0.05,
        invaild_value: int = 0, max_disp: int = 192) -> tensor:
    mask = (gt != invaild_value) & (gt < max_disp)
    mask.detach_()
    acc_res = []
    with torch.no_grad():
        total_num = mask.int().sum()
        error = torch.abs(res[mask] - gt[mask])
        related_threshold = gt[mask] * relted_error
        for i in range(threshold_num):
            threshold = start_threshold + i
            acc = (error > threshold) & (error > related_threshold)
            acc_num = acc.int().sum()
            error_rate = acc_num / (total_num + ACC_EPSILON)
            acc_res.append(error_rate)
        mae = error.sum() / (total_num + ACC_EPSILON)
    return acc_res, mae


def read_pfm(filename: str)->tuple:
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


class Evalution(nn.Module):
    """docstring for Evalution"""

    def __init__(self, start_threshold: int = 2,
                 threshold_num: int = 4, relted_error: float = 0.05,
                 invaild_value: int = 0):
        super().__init__()
        self._start_threshold = start_threshold
        self._threshold_num = threshold_num
        self._relted_error = relted_error
        self._invaild_value = invaild_value

    def forward(self, res, gt):
        return d_1(res, gt, self._start_threshold,
                   self._threshold_num, self._relted_error,
                   self._invaild_value)


def get_data(img_path: str, gt_path: str) -> np.array:
    img = np.array(Image.open(img_path), dtype=np.float32)/float(DEPTH_DIVIDING)

    file_type = os.path.splitext(gt_path)[-1]
    if file_type == ".png":
        img_gt = np.array(Image.open(gt_path), dtype=np.float32)/float(DEPTH_DIVIDING)
    else:
        img_gt, _ = read_pfm(gt_path)

    return img, img_gt


def data2cuda(img: np.array, img_gt: np.array) -> torch.tensor:
    img = torch.from_numpy(img).float()
    img_gt = torch.from_numpy(img_gt.copy()).float()

    img = Variable(img, requires_grad=False)
    img_gt = Variable(img_gt, requires_grad=False)
    return img, img_gt


def print_total(total: np.array, err_total: int,
                total_img_num: int, threshold_num: int) -> None:
    offset = 1
    str_data = 'total '
    for j in range(threshold_num):
        d1_str = '%dpx: %f ' % (j + offset, total[j] / total_img_num)
        str_data += d1_str
    str_data += 'mae: %f' % (err_total / total_img_num)
    print(str_data)


def cal_total(id_num: int, total: np.array, err_total: int,
              acc_res: torch.tensor, mae: torch.tensor,
              threshold_num: int) -> None:
    str_data = str(id_num) + ' '
    for i in range(threshold_num):
        d1_res = acc_res[i].cpu()
        d1_res = d1_res.detach().numpy()
        total[i] = total[i] + d1_res
        str_data = str_data + str(d1_res) + ' '

    mae_res = mae.cpu()
    mae_res = mae_res.detach().numpy()
    err_total += mae_res

    str_data = str_data + str(mae_res)
    print(str_data)

    return total, err_total


def main():
    # setting
    img_path_format = './ResultImg/%06d_10.png'
    gt_list_path = './Datasets/scene_flow_testing_list.csv'
    # gt_list_path = './Datasets/kitti2012_training_list.csv'
    # gt_list_path = './Datasets/kitti2015_training_list.csv'
    gt_dsp_path = read_label_list(gt_list_path)
    # total_img_num = 194
    total_img_num = len(gt_dsp_path)

    start_threshold = 1
    threshold_num = 5

    # Variable
    total = np.zeros(threshold_num)
    err_total = 0

    # push model to CUDA
    eval_model = Evalution(start_threshold=start_threshold,
                           threshold_num=threshold_num)
    eval_model = torch.nn.DataParallel(eval_model).cuda()

    for i in range(total_img_num):
        img_path = img_path_format % (i)
        gt_path = gt_dsp_path[i]

        img, img_gt = get_data(img_path, gt_path)
        img, img_gt = data2cuda(img, img_gt)

        acc_res, mae = eval_model(img, img_gt)
        total, err_total = cal_total(i, total,
                                     err_total, acc_res,
                                     mae, threshold_num)

    print_total(total, err_total, total_img_num, threshold_num)


if __name__ == '__main__':
    main()
