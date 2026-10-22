from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.models import Sequential

from config import IMAGE_SHAPE


def build_model(class_count: int):
    model = Sequential([
        Input(shape=IMAGE_SHAPE, name="input"),
        Conv2D(3, (3, 3), activation="relu", padding="same"),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPool2D((3, 3)),
        Conv2D(256, (3, 3), activation="relu"),
        Dropout(0.2),
        Conv2D(256, (3, 3), strides=2, activation="relu", padding="same"),
        MaxPool2D((2, 2)),
        Conv2D(512, (3, 3), activation="relu", padding="same"),
        Dropout(0.2),
        Conv2D(1024, (2, 2), activation="relu", padding="same"),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(1024, activation="selu"),
        Dense(class_count, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
