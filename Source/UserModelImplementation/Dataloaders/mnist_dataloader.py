# -*- coding: utf-8 -*-
import torch.nn.functional as F

import JackFramework as jf
import UserModelImplementation.user_define as user_def

import time
from torchvision import datasets, transforms
import torchvision


class MNISTDataloader(jf.UserTemplate.DataHandlerTemplate):
    """docstring for DataHandlerTemplate"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        self.__imgs_num = 0
        self.__start_time = 0

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        args = self.__args
        # return dataset
        self.__train_dataset = torchvision.datasets.MNIST(
            'MNIST', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (user_def.MIST_MEAN,), (user_def.MIST_VAR,))
            ]))
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        # return dataset
        args = self.__args
        # return dataset
        self.__val_dataset = torchvision.datasets.MNIST(
            'MNIST', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (user_def.MIST_MEAN,), (user_def.MIST_VAR,))
            ]))
        return self.__val_dataset

    def split_data(self, batch_data: tuple, is_training: bool) -> list:
        self.__start_time = time.time()
        if is_training:
            # return input_data_list, label_data_list
            return [batch_data[0]], [batch_data[1]]
            # return input_data, supplement
        return [batch_data[0]], [batch_data[1]]

    def show_train_result(self, epoch: int, loss:
                          list, acc: list,
                          duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss:
                        list, acc: list,
                        duration: float) -> None:
        assert len(loss) == len(acc)  # same model number
        info_str = self.__result_str.training_result_str(epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: list, supplement: list,
                    img_id: int, model_id: int) -> None:
        assert self.__train_dataset is not None
        args = self.__args

    def show_intermediate_result(self, epoch: int,
                                 loss: list, acc: list) -> str:
        assert len(loss) == len(acc)  # same model number
        return self.__result_str.training_intermediate_result(epoch, loss[0], acc[0])
