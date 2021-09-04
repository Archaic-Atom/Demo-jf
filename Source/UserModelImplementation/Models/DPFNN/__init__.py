# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
import UserModelImplementation.user_define as user_def


class Debug(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> list:
        args = self.__args
        # return model
        return []

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        # return opt and sch
        return [], []

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        pass

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        args = self.__args
        # return output
        return []

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        args = self.__args
        return []

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args
        return []
