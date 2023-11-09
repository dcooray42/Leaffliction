from Augmentation import balance_augmentation
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
import pickle
from PIL import Image
import plotly.express as px
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Input, MaxPooling2D, Conv2D, concatenate, Flatten, Dense
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
                        data[image_names[key]].append(np.array(img_data.resize((256, 256))))
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


#def read_dataset(path):
#    try:
#        df = pd.DataFrame(columns=image_names.values())
#        target = pd.DataFrame(columns=["target"])
#        multiple_sub_dir = True
#        data = FolderData(path)
#        count_files_folder(data)
#        sub_dir = data.get_sub_dir()
#        for dir in sub_dir:
#            dir_path = dir.get_path()
#            if (not dir_path.endswith("Apple")
#               and not dir_path.endswith("Grape")):
#                multiple_sub_dir = False
#        if multiple_sub_dir:
#            for dir in sub_dir:
#                x, y = transform_image_folder(dir)
#                df = pd.concat([df, x], ignore_index=True)
#                target = pd.concat([target, y], ignore_index=True)
#        else:
#            x, y = transform_image_folder(data)
#            df = pd.concat([df, x], ignore_index=True)
#            target = pd.concat([target, y], ignore_index=True)
#        print(df.shape)
#        print(target.shape)
#        print(target["target"].value_counts())
#    #    print(df.head(10))
##        pkl_data = {
##            "x" : df,
##            "y" : target
##        }
##        with open("data.pkl", "wb") as f:
##            pickle.dump(pkl_data, f)
#    except Exception as e:
#        raise e
#    print(1)
#    print(df.dtypes)
#    return df / 255, target

def read_dataset(path, batch_size=64):
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="both"
        )
    return dataset
    

#def create_cnn(num_classes):
#
##    def create_single_cnn(num_classes, height=256, width=256, channels=3):
#
#
#    # Define input layers for each modality
#    print(2)
#    img_height = 256
#    img_width = 256
#    img_chan = 3
#    input_original = Input(shape=(img_height, img_width, img_chan))
#    input_blur = Input(shape=(img_height, img_width, img_chan))
#    input_mask = Input(shape=(img_height, img_width, img_chan))
#    input_roi = Input(shape=(img_height, img_width, img_chan))
#    input_analyze = Input(shape=(img_height, img_width, img_chan))
#    input_pseudo = Input(shape=(img_height, img_width, img_chan))
#    input_histo = Input(shape=(img_height, img_width, img_chan))
#
#    # Define separate convolutional layers for each modality
#    conv_original = Conv2D(64, (3, 3))(input_original)
#    conv_blur = Conv2D(64, (3, 3))(input_blur)
#    conv_mask = Conv2D(64, (3, 3))(input_mask)
#    conv_roi = Conv2D(64, (3, 3))(input_roi)
#    conv_analyze = Conv2D(64, (3, 3))(input_analyze)
#    conv_pseudo = Conv2D(64, (3, 3))(input_pseudo)
#    conv_histo = Conv2D(64, (3, 3))(input_histo)
#
#    # Merge features from different channels
#    merged = concatenate([conv_original,
#                          conv_blur,
#                          conv_mask,
#                          conv_roi,
#                          conv_analyze,
#                          conv_pseudo,
#                          conv_histo], axis=-1)
#
#    # Continue with fully connected layers and output layer for classification
#    flatten = Flatten()(merged)
#    dense = Dense(128, activation="relu")(flatten)
#    output = Dense(num_classes, activation="softmax")(dense)
#
#    model = tf.keras.Model(inputs=[input_original,
#                                   input_blur,
#                                   input_mask,
#                                   conv_roi,
#                                   conv_analyze,
#                                   conv_pseudo,
#                                   conv_histo], outputs=output)
#    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#    print(3)
#    return model


def create_cnn(num_classes):
    model = Sequential()
    model.add(Rescaling(1.0 / 255))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (1, 1), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def train(model, dataset):
    print(4)
    history = model.fit(dataset[0], epochs=10, validation_data=dataset[1])
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label = "val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()

    test_loss, test_acc = model.evaluate(dataset[1], verbose=2)
    print(test_acc)


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
#    args["path"] = args["path"].rstrip("/")
#    args["dest"] = args["path"] + "/augmented_data"
#    balance_augmentation(**args)
#    args["path"], args["dest"] = args["dest"], args["path"] + "/dataset"
    del args["n_images_subfolder"]
#    balance_transformation(**args)
#    args["path"] = args["dest"]
#    del args["dest"]
    dataset = read_dataset(**args)
    print(set(dataset[0].class_names).union(set(dataset[1].class_names)))
    model = create_cnn(len(set(dataset[0].class_names).union(set(dataset[1].class_names))))
    train(model, dataset)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
