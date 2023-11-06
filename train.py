from Augmentation import balance_augmentation
from argparse import ArgumentParser
import copy
from Data import FolderData
from Distribution import count_files_folder
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from plantcv import plantcv as pcv
import pickle
from PIL import Image
import plotly.express as px
from tqdm import tqdm
from Transformation import balance_transformation


image_names = {
    "Original" : "Original",
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
                file_found = False
                try:
                    img_data = Image.open(path_file).convert("RGB")
                except Exception as e:
                    raise e
                for key in image_names.keys():
                    if key in file_name:
                        data[image_names[key]].append(np.array(img_data.convert("RGB")))
                        file_found = True
                        break
                if file_found == False:
                    raise Exception("error: feature not found")
    df = pd.DataFrame(data)
    target = pd.DataFrame({"target" : y})
    print(df.shape)
    print(target.shape)
    print(target["target"].value_counts())
    return df, target


def read_dataset(path):
    try:
        df = pd.DataFrame(columns=image_names.values())
        target = pd.DataFrame(columns=["target"])
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
                x, y = transform_image_folder(dir)
                df = pd.concat([df, x], ignore_index=True)
                target = pd.concat([target, y], ignore_index=True)
        else:
            x, y = transform_image_folder(data)
            df = pd.concat([df, x], ignore_index=True)
            target = pd.concat([target, y], ignore_index=True)
        print(df.shape)
        print(target.shape)
        print(target["target"].value_counts())
        print(df.head(10))
        pkl_data = {
            "x" : df,
            "y" : target
        }
        with open("data.pkl", "wb") as f:
            pickle.dump(pkl_data, f)
    except Exception as e:
        raise e


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder or file to transform")
    parser.add_argument("n_images_subfolder",
                         type=int,
                         help="Number of image to select in each subfolders")
#    parser.add_argument("dest",
#                        type=str,
#                        help="Path of the folder where to save the images")
    args = parser.parse_args()
#    try:
    args = vars(args)
    args["path"] = args["path"].rstrip("/")
    args["dest"] = args["path"] + "/augmented_data"
    balance_augmentation(**args)
    args["path"], args["dest"] = args["dest"], args["path"] + "/dataset"
    del args["n_images_subfolder"]
    balance_transformation(**args)
    args["path"] = args["dest"]
    del args["dest"]
    read_dataset(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
