# -*- coding: utf-8 -*
import JackFramework as jf

from .mnist_dataloader import MNISTDataloader


def dataloaders_zoo(args: object, name: str) -> object:
    for case in jf.Switch(name):
        if case('mnist'):
            jf.log.info("Enter the mnist dataloader")
            dataloader = MNISTDataloader(args)
            break
        if case(''):
            dataloader = None
            jf.log.error("The dataloader's name is error!!!")
    return dataloader
