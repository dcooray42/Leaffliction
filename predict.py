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
#    print(pcv.image_info(img)["size"])
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=85, object_type='light')
    pcv.print_image(img=s_thresh, filename="s_thresh.jpg")
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    gaussian_img = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)
    pcv.print_image(img=gaussian_img, filename="gaussian_img.jpg")
    b = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    b_thresh = pcv.threshold.binary(gray_img=b, threshold=121, 
                                object_type='light')
    pcv.print_image(img=b_thresh, filename="b_thresh.jpg")
    bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)
    masked = pcv.apply_mask(img=img, mask=bs, mask_color='white')
    pcv.print_image(img=masked, filename="masked.jpg")
    masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
    pcv.print_image(img=masked_a, filename="masked_a.jpg")
    masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')
    pcv.print_image(img=masked_b, filename="masked_b.jpg")
    maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=127, 
                                      object_type='dark')
    pcv.print_image(img=maskeda_thresh, filename="maskeda_thresh.jpg")
    maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135, 
                                       object_type='light')
    pcv.print_image(img=maskeda_thresh1, filename="maskeda_thresh1.jpg")
    maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128, 
                                      object_type='light')
    pcv.print_image(img=maskedb_thresh, filename="maskedb_thresh.jpg")
    ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)
    pcv.print_image(img=ab, filename="ab.jpg")
    xor_img = pcv.logical_xor(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
    xor_img_color = pcv.apply_mask(img=img, mask=xor_img,
                                       mask_color="white")
    ab_fill = pcv.fill(bin_img=ab, size=200)
    pcv.print_image(img=ab_fill, filename="ab_fill.jpg")
    masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')
    pcv.print_image(img=xor_img_color, filename="xor.jpg")
    pcv.print_image(img=masked2, filename="masked2.jpg")
    roi = pcv.roi.rectangle(img=masked2, x=0, y=0, h=256, w=256)
    filtered_mask = pcv.roi.filter(ab_fill, roi, roi_type="partial")
    total_mask = pcv.apply_mask(img=img, mask=filtered_mask, mask_color='white')
    pcv.print_image(img=total_mask, filename="mask.jpg")
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
