from argparse import ArgumentParser
from Data import FolderData
import matplotlib.pyplot as plt
import os
from PIL import Image


def count_files_folder(data):
    path = data.get_path()
    count = 0
    for path_file in os.listdir(path):
        complete_path = os.path.join(path, path_file)
        if os.path.isdir(complete_path):
            sub_dir_data = FolderData(complete_path)
            data.add_sub_dir(sub_dir_data)
            count_files_folder(sub_dir_data)
            data.add_count(sub_dir_data.get_count())
        else:
            try:
                Image.open(complete_path)
                count += 1
            except Exception:
                continue
    data.add_count(count)


def distribution(path):

    def print_info(data):
        print(data)
        for sub in data.get_sub_dir():
            print_info(sub)

    data = FolderData(path)
    count_files_folder(data)
    print_info(data)
    print(path)
    return


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder to analyze")
    args = parser.parse_args()
#    try:
    args = vars(args)
    distribution(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
