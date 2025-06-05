import tensorflow as tf
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from tensorflow import keras

from genre_model.config.core import config


def prepare_model():
    features_size = len(config.ml_model_config.features)
    model = Sequential(
        [
            Dense(512, activation="relu", input_shape=(features_size,)),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.Accuracy()],
    )
    return model


genre_classifier = KerasClassifier(prepare_model, epochs=100, batch_size=128)
