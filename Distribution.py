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
            folder_data.append(count_files_folder(complete_path))
        else:
            try:
                Image.open(complete_path)
                count += 1
            except Exception:
                continue
    return folder_data if len(folder_data) else FolderData(path, count)


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
