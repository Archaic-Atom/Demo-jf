# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
import UserModelImplementation.user_define as user_def


class GwcNet(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""
    MODEL_ID = 0  # only GWC-Net

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__criteria = nn.CrossEntropyLoss()
        self.__acc = None

    def get_model(self) -> list:
        args = self.__args
        # return output
        model = jf.sm.GwcNet(args.dispNum)

        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        args = self.__args
        # return opt
        opt = optim.AdamW(model[0].parameters(), lr=lr, weight_decay=1e-4)

        max_warm_up_epoch = 5
        convert_epoch = 30
        off_set = 1
        lr_factor = 1.0

        if args.lr_scheduler:
            def lr_lambda(epoch): return ((epoch + off_set) / max_warm_up_epoch) \
                if epoch < max_warm_up_epoch else lr_factor if (
                epoch >= max_warm_up_epoch and epoch < convert_epoch) else lr_factor * 0.1

            sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        else:
            sch = None

            # = sch = None
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        if self.MODEL_ID == sch_id:
            sch.step()

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        disp_0 = None
        if self.MODEL_ID == model_id:
            # print(input_data[0].size())
            disp_0, disp_1, disp_2, disp_3 = model(input_data[0], input_data[1])
        # print(pred0)
        return [disp_0, disp_1, disp_2, disp_3]

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        # args = self.__args
        acc_0 = None
        mae_0 = None

        if self.MODEL_ID == model_id:
            acc_0, mae_0 = jf.Accuracy.d_1(output_data[2], label_data[0])

        return [acc_0[1], mae_0]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        total_loss = None

        if self.MODEL_ID == model_id:
            args = self.__args
            loss_0 = jf.Loss.smooth_l1(
                output_data[0], label_data[0],
                args.startDisp, args.startDisp + args.dispNum)
            loss_1 = jf.Loss.smooth_l1(
                output_data[1], label_data[0],
                args.startDisp, args.startDisp + args.dispNum)
            loss_2 = jf.Loss.smooth_l1(
                output_data[2], label_data[0],
                args.startDisp, args.startDisp + args.dispNum)
            loss_3 = jf.Loss.smooth_l1(
                output_data[3], label_data[0],
                args.startDisp, args.startDisp + args.dispNum)
            total_loss = loss_0 + loss_1 + loss_2 + loss_3

        return [total_loss, loss_0]
