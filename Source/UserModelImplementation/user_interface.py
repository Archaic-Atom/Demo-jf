# -*- coding: utf-8 -*-
import JackFramework as jf
import argparse

import UserModelImplementation.user_define as user_def

# model
from UserModelImplementation.Models.Debug.inference import Debug
from UserModelImplementation.Models.PSMNet.inference import PsmNet
from UserModelImplementation.Models.GwcNet.inference import GwcNet
from UserModelImplementation.Models.your_model.inference import YourModel

# dataloader
from UserModelImplementation.Dataloaders.stereo_dataloader import StereoDataloader
from UserModelImplementation.Dataloaders.your_dataloader import YourDataloader


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        name = args.modelName
        for case in jf.Switch(name):
            if case('PsmNet'):
                jf.log.info("Enter the PsmNet model")
                model = PsmNet(args)
                dataloader = StereoDataloader(args)
                break
            if case('GwcNet'):
                jf.log.info("Enter the GwcNet model")
                model = GwcNet(args)
                dataloader = StereoDataloader(args)
                break
            if case('Debug'):
                jf.log.warning("Enter the debug model!!!")
                model = Debug(args)
                dataloader = StereoDataloader(args)
                break
            if case('YourModel'):
                jf.log.warning("Enter the YourModel model!")
                model = YourModel(args)
                dataloader = YourDataloader(args)
            if case():
                model = None
                dataloader = None
                jf.log.error("The model's name is error!!!")

        return model, dataloader

    def user_parser(self, parser: object) -> object:
        parser.add_argument('--startDisp', type=int,
                            default=user_def.START_DISP,
                            help='start disparity')
        parser.add_argument('--dispNum', default=user_def.DISP_NUM,
                            help='disparity number')
        parser.add_argument('--lr_scheduler', type=UserInterface.__str2bool,
                            default=user_def.LR_SCHEDULER,
                            help='use or not use lr scheduler')
        return parser

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
