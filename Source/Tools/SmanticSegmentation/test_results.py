# -*- coding: utf-8 -*-
import os
import tifffile
import glob
import numpy as np


def mean(res_path: str, gt_path: str, file_format: str):
    files = glob.glob(res_path + file_format)
    print(len(files))
    for file in files:
        name = os.path.basename(file)
        res_img = tifffile.imread(file)
        gt_img = tifffile.imread(gt_path+name)
        print(res_img)
        print(gt_img)
        res = np.mean(res_img - gt_img)
        print(res)
        return res


def main():
    res_path = '/home2/raozhibo/Documents/Programs/MSDNet/ResultImg_v2/'
    gt_path = '/home2/raozhibo/Documents/Programs/' +\
        'dfc2021-msd-baseline/results/unet_both_baseline/submission/'
    file_format = '*.tif'
    mean(res_path, gt_path, file_format)


if __name__ == "__main__":
    main()
