from argparse import ArgumentParser
from Data import FolderData
from Distribution import count_files_folder
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from plantcv import plantcv as pcv
from PIL import Image
import plotly.express as px
from tqdm import tqdm


image_names = {
    "GaussianBlur" : "Gaussian Blur",
    "Mask" : "Mask",
    "ROI" : "Roi Objects",
    "AnalyzeObject" : "Analyze object",
    "Pseudolandmarks" : "Pseudolandmarks",
    "Histogram" : "Histogram"
}


image_class = {
    "Apple_Black_rot" : 0,
    "Apple_healthy" : 1,
    "Apple_rust" : 2,
    "Apple_scab" : 3,
    "Grape_Black_rot" : 4,
    "Grape_Esca" : 5,
    "Grape_healthy" : 6,
    "Grape_spot" : 7
}


def transform_image_folder(dir):
    data = {
        "Original" : [],
        "Gaussian Blur" : [],
        "Mask" : [],
        "Roi Objects" : [],
        "Analyze object" : [],
        "Pseudolandmarks" : [],
        "Histogram" : []
    }
    y = []
    for sub_dir in dir.get_sub_dir():
        print(sub_dir)
        num_class = image_class[sub_dir.get_path().split("/")[-1]]
        for folder in tqdm(sub_dir.get_sub_dir()):
            y.append(num_class)
            for path_file in glob.glob(folder.get_path() + "/*.JPG"):
                file_name = path_file.split("/")[-1]
                try:
                    img_data = Image.open(path_file).convert("RGB")
                except Exception as e:
                    raise e
                for key in image_names.keys():
                    if key in file_name:
                        data[image_names[key]].append(img_data)
                    else:
                        data["Original"].append(img_data)
                for info in data.keys():
                    print(info, len(data[info]))
    df = pd.DataFrame(data)
    target = pd.DataFrame({"target" : y})
    print(df.shape)
    print(target.shape)
    print(target["target"].value_counts())
                        


def read_dataset(path):
    try:
        multiple_sub_dir = True
        data = FolderData(path)
        count_files_folder(data)
        sub_dir = data.get_sub_dir()
        for dir in sub_dir:
            dir_path = dir.get_path()
            if (not dir_path.endswith("Apple")
               and not dir_path.endswith("Grape")):
                multiple_sub_dir = False
        if multiple_sub_dir:
            for dir in sub_dir:
                transform_image_folder(dir)
        else:
            transform_image_folder(data)
    except Exception as e:
        raise e


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder or file to transform")
#    parser.add_argument("dest",
#                        type=str,
#                        help="Path of the folder where to save the images")
    args = parser.parse_args()
#    try:
    args = vars(args)
    args["path"] = args["path"].rstrip("/")
#        args["dest"] = args["dest"].rstrip("/")
    read_dataset(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
