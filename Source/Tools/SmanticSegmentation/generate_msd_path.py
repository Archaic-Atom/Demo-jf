# -*- coding: utf-8 -*-
import os
import glob
import linecache


def open_file(path: str, is_continue: bool = False) -> object:
    if not is_continue and os.path.exists(path):
        os.remove(path)

    return open(path, 'a')


def output_data(outputFile: object, data: str)->None:
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def get_file_list(path: str, file_format: str)->object:
    files = glob.glob(path + file_format)
    num = len(files)
    print('Number of images = ', num)
    if num == 0:
        print("No matching files found")

    return files


def get_line(filename: str, line_num: int)-> str:
    path = linecache.getline(filename, line_num)
    path = path.rstrip("\n")
    return path


def get_file_id(file_path: str, file_format: str) -> str:
    # print(file_path)
    off_set = 1
    file_name = os.path.basename(file_path)
    pos = file_name.rfind(file_format[off_set:len(file_format)])
    # print(file_id)
    return file_name[0:pos]


def generate_data_list(image_fn: str, label_fn: str, group: int) -> str:
    return image_fn + "," + label_fn + "," + str(group)


def generate_table_head(fd_training_files: object)-> None:
    data_str = generate_data_list('image_fn', 'label_fn', 'group')
    output_data(fd_training_files, data_str)


def get_val_list(path: str) -> list:
    return linecache.getlines(path)


def generate_msd_training_list(data_path: str, save_path: str,
                               image_file_format: str, label_file_format: str,
                               group: int, val_list_path: list) -> None:
    fd_training_files = open_file(save_path, False)
    files = get_file_list(data_path, image_file_format)
    generate_table_head(fd_training_files)

    val_list = None
    if val_list_path is not None:
        val_list = get_val_list(val_list_path)

    for i in range(len(files)):
        file = files[i]
        file_id = get_file_id(file, image_file_format)

        if val_list is not None and (file_id + '\n') not in val_list:
            continue

        label_fn = data_path + label_file_format % file_id
        data_str = generate_data_list(file, label_fn, group)
        output_data(fd_training_files, data_str)

    print("Finish!!!")


def main():
    root_path = "/home4/datasets/jack/MSD/"
    save_path = "./Datasets/msd_val_list_2013.csv"
    image_file_format = "*_naip-2013.tif"
    label_file_format = "%s_nlcd-2013.tif"
    group = 0
    val_list_path = './Datasets/val_tiles.txt'
    #val_list = None

    generate_msd_training_list(root_path, save_path,
                               image_file_format, label_file_format,
                               group, val_list_path)


# execute the main function
if __name__ == "__main__":
    main()
