# -*coding: utf-8 -*-
import os


# define sone struct
ROOT_PATH = '/home2/dataset/jack/Documents_home1/Database/'  # root path

# the file's path and format
RAW_DATA_FOLDER = 'Kitti/training/%s/'
LEFT_FOLDER = 'image_2'
RIGHT_FOLDER = 'image_3'
LABLE_FOLDER = 'disp_occ_0'
FILE_NAME = '%06d_10'

# file type
RAW_DATA_TYPE = '.png'
LABEL_TYPE = '.png'

# the output's path,
TRAIN_LIST_PATH = './Datasets/kitti2015_training_list.csv'
VAL_TRAINLIST_PATH = './Datasets/kitti2015_val_list.csv'

IMG_NUM = 200  # the dataset's total image
TIMES = 5     # the sample of val


def gen_raw_path(file_folder: str, num: int) -> str:
    return ROOT_PATH + RAW_DATA_FOLDER % file_folder + FILE_NAME % num + \
        RAW_DATA_TYPE


def open_file()->object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)
    if os.path.exists(VAL_TRAINLIST_PATH):
        os.remove(VAL_TRAINLIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')
    fd_val_train_list = open(VAL_TRAINLIST_PATH, 'a')

    data_str = "left_img,right_img,gt_disp"
    output_data(fd_train_list, data_str)
    output_data(fd_val_train_list, data_str)

    return fd_train_list, fd_val_train_list


def output_data(output_file: object, data: str)->None:
    output_file.write(str(data) + '\n')
    output_file.flush()


def gen_list(fd_train_list, fd_val_train_list):
    total = 0
    for num in range(IMG_NUM):

        raw_left_path = gen_raw_path(LEFT_FOLDER, num)
        raw_right_path = gen_raw_path(RIGHT_FOLDER, num)
        lable_path = gen_raw_path(LABLE_FOLDER, num)

        raw_left_path_is_exists = os.path.exists(raw_left_path)
        raw_right_path_is_exists = os.path.exists(raw_right_path)
        lable_path_is_exists = os.path.exists(lable_path)

        if (not raw_left_path_is_exists) and \
                (not lable_path_is_exists) and (not raw_right_path_is_exists):
            break

        data_str = raw_left_path + ',' + raw_right_path + ',' + lable_path
        if num % TIMES == 0:
            output_data(fd_val_train_list, data_str)
        else:
            output_data(fd_train_list, data_str)

        total += 1

    return total


def main()-> None:
    fd_train_list, fd_val_train_list = open_file()
    total = gen_list(fd_train_list, fd_val_train_list)
    print(total)


if __name__ == '__main__':
    main()