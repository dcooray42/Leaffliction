from argparse import ArgumentParser
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from PIL import Image
from plantcv import plantcv as pcv
#from tensorflow.keras.models import load_model
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
    masked_a = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    masked_b = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    kernel = np.ones((25, 25), np.uint8)
    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=125,
                                          object_type='dark')
    maskeda_thresh = pcv.fill(bin_img=maskeda_thresh, size=100)
    try:
        maskeda_thresh = pcv.fill_holes(bin_img=maskeda_thresh)
    except:
        pass
    maskeda_thresh = cv2.morphologyEx(maskeda_thresh, cv2.MORPH_CLOSE, kernel)
    pcv.print_image(maskeda_thresh, "maskeda_thresh.JPG")
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=140,
                                          object_type="light")
    maskedb_thresh = pcv.fill(bin_img=maskedb_thresh, size=100)
    try:
        maskedb_thresh = pcv.fill_holes(bin_img=maskedb_thresh)
    except:
        pass
    maskedb_thresh = cv2.morphologyEx(maskedb_thresh, cv2.MORPH_CLOSE, kernel)
    pcv.print_image(maskedb_thresh, "maskedb_thresh.JPG")
    xor_filter_mask = pcv.logical_xor(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    pcv.print_image(xor_filter_mask, "xor_filter_mask.JPG")
    or_filter_mask_1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=xor_filter_mask)
    or_filter_mask_2 = pcv.logical_or(bin_img1=maskedb_thresh, bin_img2=xor_filter_mask)
    inter_final_mask_1 = pcv.logical_and(bin_img1=or_filter_mask_1, bin_img2=maskedb_thresh)
    inter_final_mask_2 = pcv.logical_and(bin_img1=or_filter_mask_2, bin_img2=maskeda_thresh)
    final_mask = pcv.logical_or(bin_img1=inter_final_mask_1, bin_img2=inter_final_mask_2)
#    c = pcv.rgb2gray_cmyk(rgb_img=img, channel='c')
#    pcv.print_image(c, "c.JPG")
#    m = pcv.rgb2gray_cmyk(rgb_img=img, channel='m')
#    pcv.print_image(m, "m.JPG")
#    maskedm_thresh = pcv.threshold.binary(gray_img=m, threshold=90,
#                                          object_type="light")
#    maskedm_thresh = pcv.fill(bin_img=maskedm_thresh, size=100)
#    pcv.print_image(maskedm_thresh, "maskedm_thresh.JPG")
#    final_mask = pcv.logical_or(bin_img1=final_mask, bin_img2=maskedm_thresh)
#    final_mask = pcv.fill(bin_img=final_mask, size=100)
#    try:
#        final_mask = pcv.fill_holes(bin_img=final_mask)
#    except:
#        pass
#    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
#    y = pcv.rgb2gray_cmyk(rgb_img=img, channel='y')
#    pcv.print_image(y, "y.JPG")
#    k = pcv.rgb2gray_cmyk(rgb_img=img, channel='k')
#    pcv.print_image(k, "k.JPG")
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
#        predicted_class = predict(img_path, model_path, label_path)
#        additional_text = f"File name : {img_path}\nPredicted class : {predicted_class}"
#        ax2.text(0.5,
#                 0.5,
#                 additional_text,
#                 ha='center',
#                 va='center',
#                 fontsize=12)
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
