# -*- coding: utf-8 -*-
import os
import glob

ROOT_PATH = '/home2/dataset/jack/Documents_home1/Database/MiddEval/MiddEval3/'

TRAIN_LIST_PATH = './Datasets/middlebury_training_H_list.csv'
VAL_TRAIN_LIST_PATH = './Datasets/middlebury_val_H_list.csv'

TRAIN_FOLDER = 'trainingH/'
TEST_FOLDER = 'testH/'

TRAIN_FOLDER_LIST = ['Adirondack', 'Jadeplant', 'MotorcycleE', 'PianoL',
                     'Playroom', 'PlaytableP', 'Shelves', 'Vintage',
                     'ArtL', 'Motorcycle', 'Piano', 'Pipes',
                     'Playtable', 'Recycle', 'Teddy']

VAL_FOLDER_LIST = ['Australia', 'Bicycle2', 'Classroom2E', 'Crusade',
                   'Djembe', 'Hoops', 'Newkuba', 'Staircase',
                   'AustraliaP', 'Classroom2', 'Computer', 'CrusadeP',
                   'DjembeL', 'Livingroom', 'Plants']


def output_data(output_file: object, data_str: str)-> None:
    output_file.write(str(data_str) + '\n')
    output_file.flush()


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
        path = ROOT_PATH + TRAIN_FOLDER + TRAIN_FOLDER_LIST[i]

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
        path = ROOT_PATH + TEST_FOLDER + VAL_FOLDER_LIST[i]

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
