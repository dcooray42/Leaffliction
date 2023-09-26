from argparse import ArgumentParser
from Data import FolderData
import matplotlib.pyplot as plt
import os
from PIL import Image


def count_files_folder(path):
    folder_data = []
    count = 0
    for path_file in os.listdir(path):
        complete_path = os.path.join(path, path_file)
        if os.path.isdir(complete_path):
            sub_folder = count_files_folder(complete_path)
            if isinstance(sub_folder, list):
                for data in sub_folder:
                    folder_data.append(data)
            else:
                folder_data.append(count_files_folder(complete_path))
        else:
            try:
                Image.open(complete_path)
                count += 1
            except Exception:
                continue
    len_data = len(folder_data)
    if len_data and count == 0:
        return folder_data
    elif len_data and count:
        return [FolderData(path, count), folder_data]
    elif len_data == 0 and count:
        return FolderData(path, count)
    else:
        return


def distribution(path):
    arr = count_files_folder(path)
    for folder in arr:
        print(folder)
    print(path)
    return


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder to analyze")
    args = parser.parse_args()
    try:
        args = vars(args)
        distribution(**args)
    except Exception as e:
        print(str(e))
        parser.print_help()


if __name__ == "__main__":
    main()
