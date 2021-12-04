# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
import UserModelImplementation.user_define as user_def

from .model import ConvNet


class ConvNetInterface(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__criterion = nn.CrossEntropyLoss()

    def get_model(self) -> list:
        args = self.__args
        model = ConvNet(user_def.CHANNELS_NUM)
        # return model
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        # return opt and sch
        optimizer = torch.optim.Adam(model[0].parameters(), lr=lr)
        return [optimizer], [None]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        # how to do schenduler
        pass

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        args = self.__args
        # return output
        outputs = model(input_data[0])
        return [outputs]

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        args = self.__args
        total = label_data[0].size(0)
        _, predicted = torch.max(output_data[0], 1)
        correct = (predicted == label_data[0]).sum()
        return [correct / total]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args
        loss = self.__criterion(output_data[0], label_data[0])
        return [loss]
