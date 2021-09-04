# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import rasterio

import JackFramework as jf
import UserModelImplementation.user_define as user_def

import time

ID_IMG_L = 0
ID_IMG_R = 1
ID_DISP = 2
# Test
ID_TOP_PAD = 2
ID_LEFT_PAD = 3
ID_NAME = 4


class StereoDataloader(jf.UserTemplate.DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        self.__imgs_num = 0
        self.__chips_num = 0
        self.__start_time = 0

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        self.__train_dataset = jf.dataset.StereoDataset(args, args.trainListPath, is_training)
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        # return dataset
        args = self.__args
        self.__val_dataset = jf.dataset.StereoDataset(args, args.valListPath, False)
        return self.__val_dataset

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [batch_data[ID_IMG_L], batch_data[ID_IMG_R]], [batch_data[ID_DISP]]
            # return input_data, supplement
        return [batch_data[ID_IMG_L], batch_data[ID_IMG_R]], \
            [batch_data[ID_TOP_PAD], batch_data[ID_LEFT_PAD], batch_data[ID_NAME]]

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        args = self.__args

        res = output_data[0].cpu()
        res = res.detach().numpy()
        batch_size, _, _ = res.shape
        top_pads = supplement[0]
        left_pads = supplement[1]
        names = supplement[2]
        ttimes = time.time() - self.__start_time

        for i in range(batch_size):
            tmp_res = res[i, :, :]
            top_pad = top_pads[i]
            left_pad = left_pads[i]
            tmp_res = self.__train_dataset.crop_test_img(tmp_res, top_pad, left_pad)

            for case in jf.Switch(args.dataset):
                if case('US3D'):
                    jf.log.error("Unsupport the US3D dataset!!!")
                    break
                if case('kitti2012') or case('kitti2015') or case('sceneflow'):
                    name = batch_size * img_id + i
                    self.__train_dataset.save_kitti_test_data(tmp_res, name)
                    break
                if case('eth3d'):
                    name = names[i]
                    self.__train_dataset.save_eth3d_test_data(tmp_res, name, ttimes)
                    break
                if case('middlebury'):
                    name = names[i]
                    self.__train_dataset.save_middlebury_test_data(tmp_res, name, ttimes)
                    break
                if case():
                    jf.log.error("The model's name is error!!!")

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        return self.__result_str.training_intermediate_result(epoch, loss[0], acc[0])
