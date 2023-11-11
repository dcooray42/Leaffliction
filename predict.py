from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling, MaxPooling2D, Conv2D, Flatten, Dense
from tensorflow import keras


def predict(img_path, model_path, label_path):
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    img = np.array(img)
    model = load_model(model_path)
    with open(label_path, "rb") as f:
        labels = pickle.load(f)["labels"]
    result = model.predict(img[None,:,:])
    predicted_class_index = np.argmax(result)
    print(labels[predicted_class_index])


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
    predict(**args)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
