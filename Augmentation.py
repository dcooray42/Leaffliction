from argparse import ArgumentParser
from Data import FolderData
from Distribution import count_files_folder
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance


image_labels = [
    "Original",
    "Rotation",
    "Blue",
    "Contrast",
    "Scaling",
    "Illumination",
    "Projective"
]


def return_image(img, img_name, dest, img_aug=""):
    underscore_occur = 2
    save_img = img.convert("RGB")
    img_name = (img_name
                if img_aug == ""
                else ("_" + img_aug + ".").join(img_name.split(".")))
    print("/".join([dest + img_name]))
    save_img.save("/".join([dest, img_name]))
    return np.array(save_img)


def rotate_image(img, img_name, dest):
    return np.array(return_image(
        img.rotate(-10, expand=True),
        img_name,
        dest,
        "Rotate"))


def blur_image(img, img_name, dest):
    return np.array(return_image(
        img.filter(ImageFilter.GaussianBlur(2)),
        img_name,
        dest,
        "Blur"))


def contrast_image(img, img_name, dest):
    contrast_img = ImageEnhance.Contrast(img)
    return np.array(return_image(
        contrast_img.enhance(2),
        img_name,
        dest,
        "Contrast"))


def zoom_image(img, img_name, dest):
    width, height = img.size
    zoom = 10.0
    new_border_x = (width * zoom) / 100
    new_border_y = (height * zoom) / 100
    zoom_img = img.crop((new_border_x,
                         new_border_y,
                         width - new_border_x,
                         height - new_border_y))
    return np.array(return_image(
        zoom_img.resize((width, height)),
        img_name,
        dest,
        "Scaling"))


def brightness_image(img, img_name, dest):
    bright_img = ImageEnhance.Brightness(img)
    return np.array(return_image(
        bright_img.enhance(1.5),
        img_name,
        dest,
        "Illumination"))


def projective_image(img, img_name, dest):

    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0,
                           -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1,
                           -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=float)
        B = np.array(pb).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    width, height = img.size
    zoom = 10.0
    new_border_x = (width * zoom) / 100
    new_border_y = (height * zoom) / 100
    src_points = [
        (0, 0),
        (width, 0),
        (width, height),
        (0, height)
    ]
    dest_points = [
        (new_border_x / 2, new_border_y),
        (width - new_border_x, new_border_y / 2),
        (width - (new_border_x * 2), height - (new_border_y * 2)),
        (new_border_x, height)
    ]
    coeffs = find_coeffs(dest_points, src_points)
    projective_img = img.transform((width, height),
                                   Image.PERSPECTIVE,
                                   coeffs,
                                   Image.BICUBIC)
    return np.array(return_image(
        projective_img,
        img_name,
        dest,
        "Projective"))


def apply_augmentation(func, images, img, img_name, dest, iter):
    if iter > 0:
        images.append(func(img, img_name, dest))
    iter -= 1
    return iter


def augmentation(path, dest, iter):
    try:
        dest_folder = Path(dest)
        dest_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    except Exception as e:
        raise e
    try:
        img_name = path.split("/")[-1]
        img = Image.open(path)
        images = []
        iter = apply_augmentation(return_image,
                                  images, img, img_name, dest, iter)
        iter = apply_augmentation(rotate_image,
                                  images, img, img_name, dest, iter)
        iter = apply_augmentation(blur_image,
                                  images, img, img_name, dest, iter)
        iter = apply_augmentation(contrast_image,
                                  images, img, img_name, dest, iter)
        iter = apply_augmentation(zoom_image,
                                  images, img, img_name, dest, iter)
        iter = apply_augmentation(brightness_image,
                                  images, img, img_name, dest, iter)
        iter = apply_augmentation(projective_image,
                                  images, img, img_name, dest, iter)
    except Exception as e:
        raise e
    return images


def show_augmentation(path, dest):
    try:
        images = augmentation(path, dest, 5)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Augmentation images of {path}")
        ax = fig.subplots(1, len(images))
        for index, image in enumerate(images):
            ax[index].imshow(image)
            ax[index].title.set_text(image_labels[index])
            ax[index].axis("off")
        plt.show()
    except Exception as e:
        raise e


def get_max_image(path):

    def get_max_image_sub_dir(dir, max_num):
        for sub_dir in dir.get_sub_dir():
            print(sub_dir)
            img_num = sub_dir.get_count()
            if img_num > max_num:
                max_num = img_num
        return max_num

    max_num = 0
    data = FolderData(path)
    count_files_folder(data)
    sub_dir = data.get_sub_dir()
    for dir in sub_dir:
        dir_path = dir.get_path()
        print(dir_path)
        if ("/Apple" not in dir_path
           and "/Grape" not in dir_path):
            raise Exception("Can't balance the images in this folder")
        else:
            if (path.endswith("Apple")
               or path.endswith("Grape")):
                max_num = get_max_image_sub_dir(data, max_num)
                break
            else:
                max_num = get_max_image_sub_dir(dir, max_num)
    return max_num


def balance_augmentation(path, dest):
    try:
        max_img = get_max_image(path)
    except Exception as e:
        raise e
    print(max_img)


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder to analyze or file")
    parser.add_argument("dest",
                        type=str,
                        help="Path of the folder where to save the images")
    args = parser.parse_args()
#    try:
    args = vars(args)
    if os.path.isfile(args["path"]):
        show_augmentation(**args)
    else:
        balance_augmentation(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
