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


def numpy_data(data, path):

    def rec_sub_dir(folder_data, np_arr):
        x_axis = []
        y_axis = []
        sub_dir = folder_data.get_sub_dir()
        if len(sub_dir):
            for dir in sub_dir:
                x_axis.append(dir.get_path().replace(path, ""))
                y_axis.append(dir.get_count())
            np_arr.append((x_axis, y_axis))
            for dir in sub_dir:
                rec_sub_dir(dir, np_arr)

    arr_data = []
    rec_sub_dir(data, arr_data)
    return arr_data


def distribution(path):
    data = FolderData(path)
    count_files_folder(data)
    graph_data = numpy_data(data, path)
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Global distribution of the images")
    subfigs = fig.subfigures(len(graph_data), 1)
    for row, subfig in enumerate(subfigs):
        x, y = graph_data[row]
        folder_name = (x[0].split("/")[-2]
                       if x[0].split("/")[-2] != ""
                       else "parent directory")
        subfig.suptitle(
            f"Distribution of the images in the folder {folder_name}"
        )
        axs = subfig.subplots(1, 2)
        labels = [_.split("/")[-1] for _ in x]
        axs[0].pie(y, labels=labels, autopct="%1.2f%%")
        bar_container = axs[1].bar(labels, y)
        axs[1].bar_label(bar_container, fmt='{:,.0f}')
        axs[1].tick_params(labelrotation=30)
    plt.show()


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
