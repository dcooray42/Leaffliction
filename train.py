from Augmentation import balance_augmentation
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dropout, Rescaling, MaxPooling2D, Conv2D, Flatten, Dense
)
from tensorflow.keras.models import load_model
from Transformation import balance_transformation


def read_dataset(path, batch_size=64, subset="both"):
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset=subset
        )
    return dataset


def create_cnn(num_classes):
    model = Sequential()
    model.add(Rescaling(1.0 / 255))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def train(model, dataset):
    history = model.fit(dataset[0], epochs=10, validation_data=dataset[1])
    model.save("model.keras")
    pkl_data = {
        "history": history.history,
        "labels": sorted(set(dataset[0].class_names)
                         .union(set(dataset[1].class_names)))
    }
    with open("history.pkl", "wb") as f:
        pickle.dump(pkl_data, f)
    test_loss, test_acc = model.evaluate(dataset[1], verbose=2)
    print(f"test loss: {test_loss}\ntest_acc: {test_acc}")


def evaluate(model_path, history_path, test_set):
    model = load_model(model_path)
    with open(history_path, "rb") as f:
        data = pickle.load(f)
    test_loss, test_acc = model.evaluate(test_set, verbose=2)
    print(f"Test Loss: {test_loss}\nTest Accuracy: {test_acc}")
    plt.plot(data["history"]["accuracy"], label="accuracy")
    plt.plot(data["history"]["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()


def main():
    parser = ArgumentParser()
    sub_parser = parser.add_subparsers(dest="action")
    train_action = sub_parser.add_parser("train")
    train_action.add_argument("path",
                              type=str,
                              help="Path of the folder or file to transform")
    evaluate_action = sub_parser.add_parser("evaluate")
    evaluate_action.add_argument("dataset_path",
                                 type=str,
                                 help="Path of the dataset")
    evaluate_action.add_argument("model_path",
                                 type=str,
                                 help="Path of the model")
    evaluate_action.add_argument("history_path",
                                 type=str,
                                 help="Path of the history")
    args = parser.parse_args()
    try:
        action = args.action
        args = vars(args)
        args.pop("action")
        if action == "train":
            args["path"] = args["path"].rstrip("/")
            args["dest"] = args["path"] + "/augmented_data"
            balance_augmentation(**args)
            args["path"], args["dest"] = (
                args["dest"],
                args["path"] + "/dataset"
            )
            balance_transformation(**args)
            dataset = read_dataset(args["dest"])
            model = create_cnn(len(set(dataset[0].class_names)
                                   .union(set(dataset[1].class_names))))
            train(model, dataset)
        elif action == "evaluate":
            args["test_set"] = dataset = read_dataset(args["dataset_path"],
                                                      subset="validation")
            args.pop("dataset_path")
            evaluate(**args)
    except Exception as e:
        print(str(e))
        parser.print_help()


if __name__ == "__main__":
    main()
