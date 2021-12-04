# -*- coding: utf-8 -*
import JackFramework as jf

from .ConvNet.inference import ConvNetInterface


def models_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('ConvNet'):
            jf.log.info("Enter the ConvNet model")
            model = ConvNetInterface(args)
            break
        if case(''):
            model = None
            jf.log.error("The model's name is error!!!")
    return model
