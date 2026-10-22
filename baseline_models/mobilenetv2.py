from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import IMAGE_SHAPE, LEARNING_RATE


def build_model(class_count: int):
    model = Sequential([
        Input(shape=IMAGE_SHAPE),
        MobileNetV2(include_top=False, input_shape=IMAGE_SHAPE),
        Flatten(),
        Dense(1024, activation="selu"),
        Dropout(0.2),
        Dense(512, activation="selu"),
        Dense(class_count, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, epsilon=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
