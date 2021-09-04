# -*- coding: utf-8 -*-
import os
import glob


# define sone struct
ROOT_PATH = '/home2/dataset/jack/Documents_home1/Database/'  # root path
FOLDER_NAME_FORMAT = '%04d/'
RAW_DATA_FOLDER = 'frames_cleanpass/TRAIN/%s/'
LABLE_FOLDER = 'disparity/TRAIN/%s/'
LEFT_FOLDER = 'left/'
RIGHT_FOLDER = 'right/'
FILE_NAME = '%s%04d'
RAW_DATA_TYPE = '.png'
LABEL_TYPE = '.pfm'
TRAIN_LIST_PATH = './Datasets/scene_flow_training_list.csv'
LABEL_LIST_PATH = './Datasets/scene_flow_training_label_list.csv'
FOLDER_NUM = 750
ID_NUM = 3


def convert_num_to_char(folder_id: int):
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


def open_file()->object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)
    if os.path.exists(LABEL_LIST_PATH):
        os.remove(LABEL_LIST_PATH)

    fd_train_list = open(TRAIN_LIST_PATH, 'a')
    fd_label_list = open(LABEL_LIST_PATH, 'a')

    data_str = "left_img,right_img,gt_disp"
    output_data(fd_train_list, data_str)
    return fd_train_list, fd_label_list


def output_data(output_file: object, data: str)->None:
    output_file.write(str(data) + '\n')
    output_file.flush()


def gen_list_flyingthing(fd_train_list: object, fd_label_list: object) -> int:
    total = 0
    for idx in range(ID_NUM):
        for folder_num in range(FOLDER_NUM):
            num = 6
            while True:
                folder_id = convert_num_to_char(idx)
                raw_left_path = gen_raw_path(folder_id, folder_num, LEFT_FOLDER, num)
                raw_right_path = gen_raw_path(folder_id, folder_num, RIGHT_FOLDER, num)
                lable_path = gen_label_path(folder_id, folder_num, LEFT_FOLDER, num)

                raw_left_path_is_exists = os.path.exists(raw_left_path)
                raw_right_path_is_exists = os.path.exists(raw_right_path)
                lable_path_is_exists = os.path.exists(lable_path)

                if (not raw_left_path_is_exists) and \
                        (not raw_right_path_is_exists) and (not lable_path_is_exists):
                    break
                data_str = raw_left_path + ',' + raw_right_path + ',' + lable_path
                output_data(fd_train_list, data_str)
                # output_data(fd_train_list, rawRightPath)
                # output_data(fd_label_list, lablePath)
                num += 1
                total += 1
    return total


def gen_list_driving(fd_train_list: object, fd_label_list: object) -> int:
    folder_list = ['15mm_focallength/scene_backwards/fast',
                   '15mm_focallength/scene_backwards/slow',
                   '15mm_focallength/scene_forwards/fast',
                   '15mm_focallength/scene_forwards/slow',
                   '35mm_focallength/scene_backwards/fast',
                   '35mm_focallength/scene_backwards/slow',
                   '35mm_focallength/scene_forwards/fast',
                   '35mm_focallength/scene_forwards/slow']
    return produce_list(folder_list, fd_train_list, fd_label_list)


def gen_list_monkey(fd_train_list: object, fd_label_list: object) -> int:
    folder_list = ['a_rain_of_stones_x2',
                   'eating_camera2_x2',
                   'eating_naked_camera2_x2',
                   'eating_x2',
                   'family_x2',
                   'flower_storm_augmented0_x2',
                   'flower_storm_augmented1_x2',
                   'flower_storm_x2',
                   'funnyworld_augmented0_x2',
                   'funnyworld_augmented1_x2',
                   'funnyworld_camera2_augmented0_x2',
                   'funnyworld_camera2_augmented1_x2',
                   'funnyworld_camera2_x2',
                   'funnyworld_x2',
                   'lonetree_augmented0_x2',
                   'lonetree_augmented1_x2',
                   'lonetree_difftex2_x2',
                   'lonetree_difftex_x2',
                   'lonetree_winter_x2',
                   'lonetree_x2',
                   'top_view_x2',
                   'treeflight_augmented0_x2',
                   'treeflight_augmented1_x2',
                   'treeflight_x2']
    return produce_list(folder_list, fd_train_list, fd_label_list)


def produce_list(folder_list: list, fd_train_list: object, fd_label_list: object) -> int:
    total = 0
    for folder in folder_list:
        img_folder_path = ROOT_PATH + RAW_DATA_FOLDER % folder
        gt_foler_path = ROOT_PATH + LABLE_FOLDER % folder

        # print img_folder_path

        left_files = glob.glob(img_folder_path + LEFT_FOLDER + '*' + RAW_DATA_TYPE)

        for j in range(len(left_files)):
            name = os.path.basename(left_files[j])
            pos = name.find('.png')
            name = name[0:pos]
            # print name

            left_img_path = left_files[j]
            right_img_path = img_folder_path + RIGHT_FOLDER + name + RAW_DATA_TYPE
            gt_img_path = gt_foler_path + LEFT_FOLDER + name + LABEL_TYPE

            raw_left_path_is_exists = os.path.exists(left_img_path)
            raw_right_path_is_exists = os.path.exists(right_img_path)
            lable_path_is_exists = os.path.exists(gt_img_path)

            if (not raw_left_path_is_exists) and \
                    (not raw_right_path_is_exists) and (not lable_path_is_exists):
                print("\"" + left_img_path + "\"" + "is not exist!!!")
                break

            data_str = left_img_path + ',' + right_img_path + ',' + gt_img_path
            output_data(fd_train_list, data_str)
            # output_data(fd_train_list, right_img_path)
            # output_data(fd_label_list, gt_img_path)
            total += 1

    return total


def main():
    fd_train_list, fd_label_list = open_file()
    flying_num = gen_list_flyingthing(fd_train_list, fd_label_list)
    print(flying_num)
    driving_num = gen_list_driving(fd_train_list, fd_label_list)
    print(driving_num)
    monkey_num = gen_list_monkey(fd_train_list, fd_label_list)
    print(monkey_num)
    # monkey_num = 0
    total = flying_num + driving_num + monkey_num

    print(total)


if __name__ == '__main__':
    main()
