from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from plantcv import plantcv as pcv
#from tensorflow.keras.models import load_model


#def predict(img_path, model_path, label_path):
#    try:
#        img = Image.open(img_path).convert("RGB").resize((256, 256))
#        img = np.array(img)
#        model = load_model(model_path)
#        with open(label_path, "rb") as f:
#            labels = pickle.load(f)["labels"]
#        result = model.predict(img[None,:,:])
#        predicted_class_index = np.argmax(result)
#        print(labels[predicted_class_index])
#    except Exception as e:
#        raise e
#    
#    return labels[predicted_class_index]


def show_original(path, ax):
    try:
        image = Image.open(path)
        ax[0].imshow(np.array(image.convert("RGB")))
        ax[0].title.set_text("Orginal")
        ax[0].axis("off")
    except Exception as e:
        raise e


def show_mask(path, ax):
    img = pcv.readimage(path)[0]
    s = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, object_type="light")
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    b = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    b_thresh = pcv.threshold.binary(gray_img=b, threshold=118, 
                                object_type="light")
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
    masked = pcv.apply_mask(img=img, mask=bs, mask_color="white")
    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=130, 
                                       object_type="light")
    xor_img = pcv.logical_xor(bin_img1=bs, bin_img2=maskeda_thresh)
    final_img = pcv.apply_mask(img=img, mask=xor_img,
                                       mask_color="white")
    pcv.print_image(img=final_img, filename="mask.jpg")
    image = Image.open("mask.jpg")
    ax[1].imshow(np.array(image.convert("RGB")))
    ax[1].title.set_text("Masked")
    ax[1].axis("off")


def show_predict(img_path):
    try:
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Transformation images of {img_path}")
        ax = fig.subplots(1, 2)
        show_original(img_path, ax)
        show_mask(img_path, ax)
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
#    try:
    args = vars(args)
    args["img_path"] = args["img_path"].rstrip("/")
    args["model_path"] = args["model_path"].rstrip("/")
    show_predict(args["img_path"])
#    predict(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
