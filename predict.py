from argparse import ArgumentParser
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from PIL import Image
from plantcv import plantcv as pcv
from tensorflow.keras.models import load_model
from Transformation import create_mask


def predict(img_path, model_path, label_path):
    try:
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        img = np.array(img)
        model = load_model(model_path)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)["labels"]
        result = model.predict(img[None, :, :])
        predicted_class_index = np.argmax(result)
    except Exception as e:
        raise e
    return labels[predicted_class_index]


def show_original(path, ax):
    try:
        image = Image.open(path)
        ax.imshow(np.array(image.convert("RGB")))
        ax.title.set_text("Orginal")
        ax.axis("off")
    except Exception as e:
        raise e


def show_mask(path, ax):
    img = pcv.readimage(path)[0]
    final_mask = create_mask(img)
    final_img = pcv.apply_mask(img=img,
                               mask=final_mask,
                               mask_color="white")
    pcv.print_image(img=final_img, filename="mask.JPG")
    file_exist = glob.glob("mask.JPG")
    if len(file_exist) == 0:
        raise Exception("error: File doesn't exist")
    if len(file_exist) > 1:
        raise Exception("error: Multiple files found")
    image = Image.open("mask.JPG")
    os.remove("mask.JPG")
    ax.imshow(np.array(image.convert("RGB")))
    ax.title.set_text("Masked")
    ax.axis("off")


def show_predict(img_path, model_path, label_path):
    try:
        fig = plt.figure(constrained_layout=True)
        fig.suptitle("===     DL Classification     ===")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.2])
        ax0 = fig.add_subplot(gs[0, 0])
        show_original(img_path, ax0)
        ax1 = fig.add_subplot(gs[0, 1])
        show_mask(img_path, ax1)
        ax2 = fig.add_subplot(gs[1, :])
        predicted_class = predict(img_path, model_path, label_path)
        additional_text = (
            f"File name: {img_path}\n"
            f"Predicted class: {predicted_class}"
        )
        ax2.text(0.5,
                 0.5,
                 additional_text,
                 ha='center',
                 va='center',
                 fontsize=12)
        ax2.axis("off")
        plt.show()
    except Exception as e:
        raise e


def main():
    parser = ArgumentParser()
    parser.add_argument("img_path",
                        type=str,
                        help="Path of the file to predict")
    parser.add_argument("model_path",
                        type=str,
                        help="Path of the trained model")
    parser.add_argument("label_path",
                        type=str,
                        help="Path of the labels")

    args = parser.parse_args()
    try:
        args = vars(args)
        args["img_path"] = args["img_path"].rstrip("/")
        args["model_path"] = args["model_path"].rstrip("/")
        show_predict(**args)
    except Exception as e:
        print(str(e))
        parser.print_help()


if __name__ == "__main__":
    main()
