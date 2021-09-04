# -*- coding: utf-8 -*-
import os
import glob


# define sone struct
TRAIN_LIST_PATH = './Datasets/dfc_training_list.csv'


def open_file() -> object:
    if os.path.exists(TRAIN_LIST_PATH):
        os.remove(TRAIN_LIST_PATH)

    return open(TRAIN_LIST_PATH, 'a')


def output_data(outputFile: object, data: str) -> None:
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def gen_DFC_list(input_folder: str) -> None:
    fd_train_list = open_file()
    # get list of files
    files = glob.glob(input_folder + '*LEFT_RGB*.tif')
    num = len(files)
    print('Number of images = ', num)
    if num == 0:
        print("No matching files found")
        # print6("No matching files found", file=stderr)
        return

    output_data(fd_train_list, "left_img,left_msi,right_img,right_msi,gt_dsp")
    for i in range(num):
        left_name = os.path.basename(files[i])
        start = left_name.find('LEFT_RGB')
        right_name = input_folder + left_name[0:start] + 'RIGHT_RGB.tif'
        left_msi_name = input_folder + left_name[0:start] + 'LEFT_MSI.tif'
        right_msi_name = input_folder + left_name[0:start] + 'RIGHT_MSI.tif'
        disparity_name = input_folder + left_name[0:start] + 'LEFT_DSP.tif'
        left_name = input_folder + left_name

        exist_0 = os.path.exists(left_name)
        exist_1 = os.path.exists(left_msi_name)
        exist_2 = os.path.exists(right_name)
        exist_3 = os.path.exists(right_msi_name)
        exist_4 = os.path.exists(disparity_name)

        if (not exist_0) or \
            (not exist_1) or \
            (not exist_2) or \
            (not exist_3) or \
                (not exist_4):
            print("'" + left_name + "' : is not existed!")
            print("'" + left_msi_name + "' : is not existed!")
            print("'" + right_name + "' : is not existed!")
            print("'" + disparity_name + "' : is not existed!")
            print('***************')
            break

        res = left_name + ',' + left_msi_name + ',' + right_name \
            + ',' + right_msi_name + ','+disparity_name
        output_data(fd_train_list, res)
        # OutputData(fd_train_list, left_msi_name)
        # OutputData(fd_train_list, right_name)
        # OutputData(fd_train_list, right_msi_name)
        # OutputData(fd_label_list, left_cls_name)
        # OutputData(fd_label_list, disparity_name)

    print("Finish!")


if __name__ == "__main__":
    gen_DFC_list("/home4/datasets/jack/Documents_home2/DFC2019_track2_trainval/Track_Train/")
