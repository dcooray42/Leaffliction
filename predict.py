from argparse import ArgumentParser
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from PIL import Image
from plantcv import plantcv as pcv
#from tensorflow.keras.models import load_model
from Transformation import create_mask
#
#
#def predict(img_path, model_path, label_path):
#    try:
#        img = Image.open(img_path).convert("RGB").resize((256, 256))
#        img = np.array(img)
#        model = load_model(model_path)
#        with open(label_path, "rb") as f:
#            labels = pickle.load(f)["labels"]
#        result = model.predict(img[None, :, :])
#        predicted_class_index = np.argmax(result)
#    except Exception as e:
#        raise e
#    return labels[predicted_class_index]


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
#    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
#    s_thresh = pcv.threshold.binary(gray_img=s,
#                                    threshold=85,
#                                    object_type="light")
#    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
#    b = pcv.rgb2gray_lab(rgb_img=img, channel="b")
#    b_thresh = pcv.threshold.binary(gray_img=b, threshold=118,
#                                    object_type="light")
#    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
#    masked = pcv.apply_mask(img=img, mask=bs, mask_color="white")
    masked_a = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    masked_b = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=125,
                                          object_type='dark')
    maskeda_thresh = pcv.fill(bin_img=maskeda_thresh, size=100)
    maskeda_thresh = pcv.fill_holes(bin_img=maskeda_thresh)
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=140,
                                          object_type="light")
    maskedb_thresh = pcv.fill(bin_img=maskedb_thresh, size=100)
    maskedb_thresh = pcv.fill_holes(bin_img=maskedb_thresh)
    xor_filter_mask = pcv.logical_xor(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    or_filter_mask_1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=xor_filter_mask)
    or_filter_mask_2 = pcv.logical_or(bin_img1=maskedb_thresh, bin_img2=xor_filter_mask)
    inter_final_mask_1 = pcv.logical_and(bin_img1=or_filter_mask_1, bin_img2=maskedb_thresh)
    inter_final_mask_2 = pcv.logical_and(bin_img1=or_filter_mask_2, bin_img2=maskeda_thresh)
    final_mask = pcv.logical_or(bin_img1=inter_final_mask_1, bin_img2=inter_final_mask_2)
    h = pcv.rgb2gray_hsv(rgb_img=img, channel="h")
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    v = pcv.rgb2gray_hsv(rgb_img=img, channel="v")
    h = pcv.threshold.binary(gray_img=h, threshold=118,
                                    object_type="dark")
    s = pcv.threshold.binary(gray_img=s, threshold=140,
                                    object_type="light")
    v = pcv.threshold.binary(gray_img=v, threshold=20,
                                    object_type="dark")
    pcv.print_image(h, "h.JPG")
    pcv.print_image(s, "s.JPG")
    pcv.print_image(v, "v.JPG")
#    mask = create_mask(img)
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
    #    predicted_class = predict(img_path, model_path, label_path)
    #    additional_text = f"File name : {img_path}\nPredicted class : {predicted_class}"
    #    ax2.text(0.5,
    #             0.5,
    #             additional_text,
    #             ha='center',
    #             va='center',
    #             fontsize=12)
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
