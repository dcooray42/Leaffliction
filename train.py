from Augmentation import balance_augmentation
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, MaxPooling2D, Conv2D, Flatten, Dense
from Transformation import balance_transformation


def read_dataset(path, batch_size=64):
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="both"
        )
    return dataset


def create_cnn(num_classes):
    model = Sequential()
    model.add(Rescaling(1.0 / 255))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (1, 1), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
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
        "history" : history,
        "labels" : sorted(set(dataset[0].class_names).union(set(dataset[1].class_names)))
    }
    with open("history.pkl", "wb") as f:
        pickle.dump(pkl_data, f)
        
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label = "val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()

    test_loss, test_acc = model.evaluate(dataset[1], verbose=2)
    print(f"test loss: {test_loss}\ntest_acc: {test_acc}")


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder or file to transform")
    parser.add_argument("--n_images_subfolder",
                        type=int,
                        default=-1,
                        help="Number of images to select in each subfolder",
                        required=False)

    args = parser.parse_args()
#    try:
    args = vars(args)
#    args["path"] = args["path"].rstrip("/")
#    args["dest"] = args["path"] + "/augmented_data"
#    balance_augmentation(**args)
#    args["path"], args["dest"] = args["dest"], args["path"] + "/dataset"
    del args["n_images_subfolder"]
#    balance_transformation(**args)
#    args["path"] = args["dest"]
#    del args["dest"]
    dataset = read_dataset(**args)
    model = create_cnn(len(set(dataset[0].class_names).union(set(dataset[1].class_names))))
    train(model, dataset)
#    except Exception as e:
#        print(str(e))
#        parser.print_help()


if __name__ == "__main__":
    main()
