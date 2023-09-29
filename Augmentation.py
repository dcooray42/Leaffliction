from argparse import ArgumentParser
from Distribution import count_files_folder
import matplotlib.pyplot as plt
import numpy as np
import os
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


def return_image(img):
    return np.array(img.convert("RGB"))


def rotate_image(img):
    return np.array(return_image(img.rotate(-10, expand=True)))


def blur_image(img):
    return np.array(return_image(
        img.filter(ImageFilter.GaussianBlur(2))))


def contrast_image(img):
    contrast_img = ImageEnhance.Contrast(img)
    return np.array(return_image(contrast_img.enhance(2)))


def zoom_image(img):
    width, height = img.size
    zoom = 10.0
    new_border_x = (width * zoom) / 100
    new_border_y = (height * zoom) / 100
    zoom_img = img.crop((new_border_x,
                         new_border_y,
                         width - new_border_x,
                         height - new_border_y))
    return np.array(return_image(
        zoom_img.resize((width, height))))


def brightness_image(img):
    bright_img = ImageEnhance.Brightness(img)
    return np.array(return_image(bright_img.enhance(1.5)))


def projective_image(img):

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
    return np.array(return_image(projective_img))


def augmentation(path):
    try:
        img = Image.open(path)
        images = []
        images.append(return_image(img))
        images.append(rotate_image(img))
        images.append(blur_image(img))
        images.append(contrast_image(img))
        images.append(zoom_image(img))
        images.append(brightness_image(img))
        images.append(projective_image(img))
    except Exception as e:
        raise e
    return images


def show_augmentation(path):
    try:
        images = augmentation(path)
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


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder to analyze")
    args = parser.parse_args()
#    try:
    args = vars(args)
    show_augmentation(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
