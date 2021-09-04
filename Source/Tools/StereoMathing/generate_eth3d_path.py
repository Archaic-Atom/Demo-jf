# -*- coding: utf-8 -*-
import os
import glob

from PIL import Image
import numpy as np

ROOT_PATH = '/home2/dataset/jack/Documents_home1/Database/ETH3D_Stereo/'
TRAIN_LIST_PATH = './Datasets/eth3d_training_list.csv'
VAL_TRAIN_LIST_PATH = './Datasets/eth3d_testing_list.csv'

TRAIN_FOLDER_LIST = ['TRAIN/delivery_area_1l', 'TRAIN/delivery_area_1s',
                     'TRAIN/delivery_area_2l', 'TRAIN/delivery_area_2s',
                     'TRAIN/delivery_area_3l', 'TRAIN/delivery_area_3s',
                     'TRAIN/electro_1l', 'TRAIN/electro_1s',
                     'TRAIN/electro_2l', 'TRAIN/electro_2s',
                     'TRAIN/electro_3l', 'TRAIN/electro_3s',
                     'TRAIN/facade_1s', 'TRAIN/forest_1s',
                     'TRAIN/forest_2s', 'TRAIN/playground_1l',
                     'TRAIN/playground_1s', 'TRAIN/playground_2l',
                     'TRAIN/playground_2s', 'TRAIN/playground_3l',
                     'TRAIN/playground_3s', 'TRAIN/terrace_1s',
                     'TRAIN/terrace_2s', 'TRAIN/terrains_1l',
                     'TRAIN/terrains_1s', 'TRAIN/terrains_2l',
                     'TRAIN/terrains_2s']

VAL_FOLDER_LIST = ['TEST/lakeside_1l', 'TEST/lakeside_1s', 'TEST/sand_box_1l',
                   'TEST/sand_box_1s', 'TEST/storage_room_1l',
                   'TEST/storage_room_1s', 'TEST/storage_room_2_1l',
                   'TEST/storage_room_2_1s', 'TEST/storage_room_2_2l',
                   'TEST/storage_room_2_2s', 'TEST/storage_room_2l',
                   'TEST/storage_room_2s', 'TEST/storage_room_3l',
                   'TEST/storage_room_3s', 'TEST/tunnel_1l',
                   'TEST/tunnel_2l', 'TEST/tunnel_2s', 'TEST/tunnel_1s',
                   'TEST/tunnel_3l', 'TEST/tunnel_3s']


def output_data(output_file: object, data_str: str) -> None:
    output_file.write(str(data_str) + '\n')
    output_file.flush()


def read_img(path: str)->np.array:
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    return img


def open_file()->object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)

    if os.path.exists(VAL_TRAIN_LIST_PATH):
        os.remove(VAL_TRAIN_LIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')
    fd_val_train_list = open(VAL_TRAIN_LIST_PATH, 'a')

    data_str = "left_img,right_img,gt_disp"
    output_data(fd_train_list, data_str)
    output_data(fd_val_train_list, data_str)
    return fd_train_list, fd_val_train_list


def gen_tranining_list(fd_train_list: object)-> None:
    for i in range(len(TRAIN_FOLDER_LIST)):
        path = ROOT_PATH + TRAIN_FOLDER_LIST[i]

        path_0 = path + '/im0.png'
        path_1 = path + '/im1.png'
        path_2 = path + '/disp0GT.pfm'

        exist_0 = os.path.exists(path_0)
        exist_1 = os.path.exists(path_1)
        exist_2 = os.path.exists(path_2)

        if (not exist_0) or \
            (not exist_1) or \
                (not exist_2):
            print("'" + path_0 + "' : is not existed!")
            print("'" + path_1 + "' : is not existed!")
            print("'" + path_2 + "' : is not existed!")
            print('***************')
            break

        data_str = path_0 + ',' + path_1 + ',' + path_2
        output_data(fd_train_list, data_str)
        print("Finish: " + TRAIN_FOLDER_LIST[i])


def gen_testing_list(fd_val_train_list: object)->None:
    for i in range(len(VAL_FOLDER_LIST)):
        path = ROOT_PATH + VAL_FOLDER_LIST[i]

        path_0 = path + '/im0.png'
        path_1 = path + '/im1.png'

        exist_0 = os.path.exists(path_0)
        exist_1 = os.path.exists(path_1)

        if (not exist_0) or \
                (not exist_1):
            print("'" + path_0 + "' : is not existed!")
            print("'" + path_1 + "' : is not existed!")
            print('***************')
            break

        data_str = path_0 + ',' + path_1 + ',None'
        output_data(fd_val_train_list, data_str)

        print("Finish: " + VAL_FOLDER_LIST[i])


def main()-> None:
    fd_train_list, fd_val_train_list = open_file()
    gen_tranining_list(fd_train_list)
    gen_testing_list(fd_val_train_list)


if __name__ == '__main__':
    main()
