from argparse import ArgumentParser
import cv2
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


image_labels = [
    "Original",
    "Gaussian Blur",
    "Mask",
    "Roi objects",
    "Analyze object",
    "Pseudolandmarks",
    "Histogram"
]


def create_mask(img):
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    s_thresh = pcv.threshold.binary(gray_img=s,
                                    threshold=85,
                                    object_type="light")
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    b = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    b_thresh = pcv.threshold.binary(gray_img=b, threshold=118,
                                    object_type="light")
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
    masked = pcv.apply_mask(img=img, mask=bs, mask_color="white")
    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")
    kernel = np.ones((25, 25), np.uint8)
    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=125,
                                          object_type='dark')
    maskeda_thresh = pcv.fill(bin_img=maskeda_thresh, size=100)
    maskeda_thresh = cv2.morphologyEx(maskeda_thresh, cv2.MORPH_CLOSE, kernel)
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=140,
                                          object_type="light")
    maskedb_thresh = pcv.fill(bin_img=maskedb_thresh, size=100)
    maskedb_thresh = cv2.morphologyEx(maskedb_thresh, cv2.MORPH_CLOSE, kernel)
    xor_filter_mask = pcv.logical_xor(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    or_filter_mask_1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=xor_filter_mask)
    or_filter_mask_2 = pcv.logical_or(bin_img1=maskedb_thresh, bin_img2=xor_filter_mask)
    inter_final_mask_1 = pcv.logical_and(bin_img1=or_filter_mask_1, bin_img2=maskedb_thresh)
    inter_final_mask_2 = pcv.logical_and(bin_img1=or_filter_mask_2, bin_img2=maskeda_thresh)
    final_mask = pcv.logical_or(bin_img1=inter_final_mask_1, bin_img2=inter_final_mask_2)
    return final_mask


def return_image(img, img_name, dest, img_aug=""):
    img_name = (img_name
                if img_aug == ""
                else ("_" + img_aug + ".").join(img_name.split(".")))
    dest_path = "/".join([dest, img_name])
    if type(img).__module__ == np.__name__:
        pcv.print_image(img, dest_path)
    else:
        img.write_image(dest_path)
    return dest_path


def gaussian_blur(img, img_name, dest):
    gaussian_img = pcv.gaussian_blur(img, (5, 5))
    return return_image(gaussian_img, img_name, dest, "GaussianBlur")


def mask_image(img, img_name, dest, mask):
    masked_image = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return return_image(masked_image, img_name, dest, "Mask")


def roi_image(img, img_name, dest, mask):
    pcv.params.debug = "print"
    pcv.roi.rectangle(img=img, x=0, y=0, h=256, w=256)
    pcv.params.debug = None
    file_exist = glob.glob("*_roi.png")
    if len(file_exist) == 0:
        raise Exception("error: File doesn't exist")
    if len(file_exist) > 1:
        raise Exception("error: Multiple files found")
    for path_file in file_exist:
        roi_img = pcv.readimage(path_file)[0]
        os.remove(path_file)
    for index_y, y in enumerate(roi_img):
        for index_x, _ in enumerate(y):
            if (mask[index_y][index_x] != 0
               and index_y > 3 and index_y < 252
               and index_x > 3 and index_x < 252):
                roi_img[index_y][index_x] = [0, 255, 0]
    return return_image(roi_img, img_name, dest, "ROI")


def analyze_object_image(img, img_name, dest, mask):
    shape_img = pcv.analyze.size(img=img, labeled_mask=mask)
    return return_image(shape_img, img_name, dest, "AnalyzeObject")


def pseudolandmarks_image(img, img_name, dest, mask):
    pcv.params.debug = "print"
    pcv.homology.x_axis_pseudolandmarks(img=img, mask=mask)
    pcv.params.debug = None
    file_exist = glob.glob("*_x_axis_pseudolandmarks.png")
    if len(file_exist) == 0:
        raise Exception("error: File doesn't exist")
    if len(file_exist) > 1:
        raise Exception("error: Multiple files found")
    for path_file in file_exist:
        pseudo_img = pcv.readimage(path_file)[0]
        os.remove(path_file)
    return return_image(pseudo_img, img_name, dest, "Pseudolandmarks")


def histogram_image(img, img_name, dest, mask):
    pcv.analyze.color(img,
                      mask,
                      n_labels=1,
                      colorspaces="all",
                      label="plant_hist")
    hist_data = pcv.outputs.observations["plant_hist1"]
    columns = ["Pixel intensity", "Proportions of pixels (%)", "color Channel"]
    df = pd.DataFrame(columns=columns)
    for key in hist_data.keys():
        if "_frequencies" in key:
            x = list(range(0, 256))
            y = hist_data[key]["value"]
            color = [key.replace("_frequencies", "") for _ in range(0, 256)]
            while len(y) < 256:
                y.append(0)
            data = {"Pixel intensity": x,
                    "Proportions of pixels (%)": y,
                    "color Channel": color}
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    fig = px.line(df,
                  x="Pixel intensity",
                  y="Proportions of pixels (%)",
                  color="color Channel")
    return return_image(fig, img_name, dest, "Histogram")


def transformation(path, dest, hist=False):
    try:
        dest_folder = Path(dest)
        dest_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    except Exception as e:
        raise e
    try:
        img_name = path.split("/")[-1]
        vis = pcv.readimage(path)[0]
        mask = create_mask(vis)
        images = []
        images.append(return_image(vis, img_name, dest, "Original"))
        images.append(gaussian_blur(mask, img_name, dest))
        images.append(mask_image(vis, img_name, dest, mask))
        images.append(roi_image(vis, img_name, dest, mask))
        images.append(analyze_object_image(vis, img_name, dest, mask))
        images.append(pseudolandmarks_image(vis, img_name, dest, mask))
        if hist is True:
            images.append(histogram_image(vis, img_name, dest, mask))
    except Exception as e:
        raise e
    return images


def show_transformation(path, dest):
    try:
        images = transformation(path, dest, True)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Transformation images of {path}")
        ax = fig.subplots(1, len(images))
        print(f"len image = {len(images)}")
        for index, image_path in enumerate(images):
            print(f"image path = {image_path}")
            image = Image.open(image_path)
            ax[index].imshow(np.array(image.convert("RGB")))
            ax[index].title.set_text(image_labels[index])
            ax[index].axis("off")
        plt.show()
    except Exception as e:
        raise e


def transform_image_folder(dir, dest, depth_folder):
    for sub_dir in dir.get_sub_dir():
        transformed_dest = dest + "/" + "/".join(
            sub_dir.get_path().split("/")[-depth_folder:])
        for path_file in tqdm(glob.glob(sub_dir.get_path() + "/*.JPG")):
            transformation(path_file,
                           transformed_dest)


def balance_transformation(path, dest):
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
                transform_image_folder(dir, dest, 1)
        else:
            transform_image_folder(data, dest, 1)
    except Exception as e:
        raise e


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder or file to transform")
    parser.add_argument("dest",
                        type=str,
                        help="Path of the folder where to save the images")
    args = parser.parse_args()
#    try:
    args = vars(args)
    args["path"] = args["path"].rstrip("/")
    args["dest"] = args["dest"].rstrip("/")
    if os.path.isfile(args["path"]):
        show_transformation(**args)
    else:
        balance_transformation(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
