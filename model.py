import numpy as np
import tensorflow as tf
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose

import os

from matplotlib import pyplot as plt

compressed_size = 32

def get_model():

    e_input_layer = keras.Input(shape=(28, 28, 1))

    layer = Conv2D(16, 3, activation="relu", strides=2, padding="same")(e_input_layer)
    conv_output = Conv2D(32, 3, activation="relu", strides=2, padding="same")(layer)
    layer = Flatten()(conv_output)
    z = Dense(compressed_size, activation="relu")(layer)

    encoder = keras.Model(e_input_layer, z, name="encoder")

    encoder.summary()
    encoder.compile(optimizer=optimizers.Adam(learning_rate=0.002), loss="mse", metrics=["mae"])

    d_input_layer = keras.Input(shape=(compressed_size,))
    layer = layers.Dense(conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3], activation="relu")(d_input_layer)
    layer = layers.Reshape(conv_output.shape[1:4])(layer)
    layer = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(layer)
    layer = Conv2DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(layer)

    decoder = keras.Model(d_input_layer, layer, name="decoder")

    decoder.summary()
    decoder.compile(optimizer=optimizers.Adam(learning_rate=0.002), loss="mse", metrics=["mae"])

    autoencoder = keras.Model(e_input_layer, decoder(z), name="auto-encoder")

    autoencoder.summary()
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.002), loss="mse", metrics=["mae"])
    return autoencoder, encoder, decoder
