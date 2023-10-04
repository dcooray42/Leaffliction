from argparse import ArgumentParser
import copy
from Data import FolderData
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from plantcv import plantcv as pcv
from PIL import Image
import plotly.express as px


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
    gray_img = pcv.rgb2gray(img)
    bin_mask = pcv.threshold.gaussian(gray_img=gray_img, ksize=250, offset=15,
                                      object_type="dark")
    rect_roi = pcv.roi.rectangle(img=img, x=0, y=0, h=256, w=256)
    cleaned_mask = pcv.fill(bin_img=bin_mask, size=50)
    return pcv.roi.filter(mask=cleaned_mask, roi=rect_roi, roi_type="partial")


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
    for path_file in glob.glob("*_roi.png"):
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
    fig = px.line(df, x="Pixel intensity", y="Proportions of pixels (%)", color="color Channel")
    return return_image(fig, img_name, dest, "Histogram")


#def brightness_image(img, img_name, dest):
#    bright_img = ImageEnhance.Brightness(img)
#    return np.array(return_image(
#        bright_img.enhance(1.5),
#        img_name,
#        dest,
#        "Illumination"))


def apply_augmentation(func, images, img, img_name, dest, iter):
    if iter > 0:
        images.append(func(img, img_name, dest))
    iter -= 1
    return iter


def transformation(path, dest):
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
        images.append(return_image(vis, img_name, dest))
        images.append(gaussian_blur(mask, img_name, dest))
        images.append(mask_image(vis, img_name, dest, mask))
        images.append(roi_image(vis, img_name, dest, mask))
        images.append(analyze_object_image(vis, img_name, dest, mask))
#        images.append(return_image(vis, img_name, dest))
        images.append(histogram_image(vis, img_name, dest, mask))
    except Exception as e:
        raise e
    return images


#def copy_original_image(path, dest, iter):
#    try:
#        dest_folder = Path(dest)
#        dest_folder.mkdir(parents=True, exist_ok=False)
#    except FileExistsError:
#        pass
#    except Exception as e:
#        raise e
#    try:
#        img_name = path.split("/")[-1]
#        img = Image.open(path)
#        return_image(img, img_name, dest)
#    except Exception as e:
#        raise e
#    return iter - 1



def show_transformation(path, dest):
    try:
        images = transformation(path, dest)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Transformation images of {path}")
        ax = fig.subplots(1, len(images))
        print(f"len image = {len(images)}")
        for index, image_path in enumerate(images):
            print(f"image path = {image_path}")
            image = Image.open(image_path)
            print(type(image))
            ax[index].imshow(np.array(image.convert("RGB")))
            ax[index].title.set_text(image_labels[index])
            ax[index].axis("off")
        plt.show()
    except Exception as e:
        raise e


#def transform_image_folder(dir, dest, depth_folder, max_num):
#    for sub_dir in dir.get_sub_dir():
#        tmp_num = max_num
#        loop = True
#        transformed_dest = dest + "/" + "/".join(
#            sub_dir.get_path().split("/")[-depth_folder:])
#        for path_file in glob.glob(sub_dir.get_path() + "/*.JPG"):
#            if tmp_num > 0:
#                tmp_num = copy_original_image(path_file,
#                                              transformed_dest,
#                                              tmp_num)
#            else:
#                loop = False
#                break
#        while loop:
#            for path_file in glob.glob(sub_dir.get_path() + "/*.JPG"):
#                if tmp_num > 0:
#                    tmp_num = augmentation(path_file,
#                                           transformed_dest,
#                                           tmp_num,
#                                           False)[1]
#                else:
#                    loop = False
#                    break


#def balance_transformation(path, dest):
#    try:
#        multiple_sub_dir = True
#        data = FolderData(path)
#        max_img = get_max_image(data, path)
#        sub_dir = data.get_sub_dir()
#        for dir in sub_dir:
#            dir_path = dir.get_path()
#            if (not dir_path.endswith("Apple")
#               and not dir_path.endswith("Grape")):
#                multiple_sub_dir = False
#        if multiple_sub_dir:
#            for dir in sub_dir:
#                transform_image_folder(dir, dest, 2, max_img)
#        else:
#            transform_image_folder(data, dest, 1, max_img)
#    except Exception as e:
#        raise e


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
#        else:
#            balance_augmentation(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
