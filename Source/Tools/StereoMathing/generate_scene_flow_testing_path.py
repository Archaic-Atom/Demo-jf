# -*- coding: utf-8 -*-
import os


# define sone struct
ROOT_PATH = '/home2/dataset/jack/Documents_home1/Database/'  # root path
FOLDER_NAME_FORMAT = '%04d/'
RAW_DATA_FOLDER = 'frames_cleanpass/TEST/%s/'
LABLE_FOLDER = 'disparity/TEST/%s/'
LEFT_FOLDER = 'left/'
RIGHT_FOLDER = 'right/'
FILE_NAME = '%s%04d'
RAW_DATA_TYPE = '.png'
LABEL_TYPE = '.pfm'
TRAIN_LIST_PATH = './Datasets/scene_flow_testing_list.csv'
FOLDER_NUM = 437
ID_NUM = 3
OFF_SET = 1


def convert_num2char(folder_id: int):
    res = 'None'
    if folder_id == 0:
        res = 'A'
    elif folder_id == 1:
        res = 'B'
    elif folder_id == 2:
        res = 'C'
    return res


def gen_raw_path(folder_id: int, folder_num: int, file_folder: str, num: int) -> str:
    return (
        ROOT_PATH
        + RAW_DATA_FOLDER % folder_id
        + FOLDER_NAME_FORMAT % folder_num
        + FILE_NAME % (file_folder, num)
        + RAW_DATA_TYPE
    )


def gen_label_path(folder_id: int, folder_num: int, file_folder: str, num: int) -> str:
    return (
        ROOT_PATH
        + LABLE_FOLDER % folder_id
        + FOLDER_NAME_FORMAT % folder_num
        + FILE_NAME % (file_folder, num)
        + LABEL_TYPE
    )


def open_file() -> object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')

    data_str = "left_img,right_img,gt_disp"
    output_data(fd_train_list, data_str)

    return fd_train_list


def output_data(output_file: object, data: str):
    output_file.write(str(data) + '\n')
    output_file.flush()


def gen_list(fd_train_list: object) -> int:
    total = 0
    for i in range(ID_NUM):
        for folder_num in range(FOLDER_NUM):
            num = 6
            while True:
                folder_id = convert_num2char(i)
                raw_left_path = gen_raw_path(folder_id, folder_num, LEFT_FOLDER, num)
                raw_right_path = gen_raw_path(folder_id, folder_num, RIGHT_FOLDER, num)
                lable_path = gen_label_path(folder_id, folder_num, LEFT_FOLDER, num)

                raw_left_pathis_exists = os.path.exists(raw_left_path)
                raw_right_pathis_exists = os.path.exists(raw_right_path)
                lable_pathis_exists = os.path.exists(lable_path)

                if (not raw_left_pathis_exists) and \
                        (not lable_pathis_exists) and (not raw_right_pathis_exists):
                    break

                data_str = raw_left_path + ',' + raw_right_path + ',' + lable_path
                output_data(fd_train_list, data_str)

                num += OFF_SET
                total += OFF_SET
    return total


def main():
    fd_train_list = open_file()
    total = gen_list(fd_train_list)
    print(total)


if __name__ == '__main__':
    main()
