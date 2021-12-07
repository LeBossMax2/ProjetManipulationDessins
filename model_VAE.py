import numpy as np
import tensorflow as tf
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose
import tensorflow.keras.backend as K

import os

from matplotlib import pyplot as plt

compressed_size = 32
lambda_loss = 1e-5

class VariationalLayer(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # KL divergence loss
        kl_batch = K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        self.add_loss(-0.5 * K.mean(kl_batch) * lambda_loss, inputs = inputs)

        # Sampling reparameterization
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_model():
    optimizer = optimizers.Adam(learning_rate=0.0005)
    loss = "binary_crossentropy"
    metrics = ["mae"]

    e_input_layer = keras.Input(shape=(28, 28, 1))

    layer = Conv2D(16, 3, activation="relu", strides=2, padding="same")(e_input_layer)
    conv_output = Conv2D(32, 3, activation="relu", strides=2, padding="same")(layer)
    layer = Flatten()(conv_output)
    layer = Dense(128, activation="relu")(layer)
    z_mean = Dense(compressed_size, name="z_mean")(layer)
    z_log_var = Dense(compressed_size, name="z_log_var")(layer)
    z = VariationalLayer()([z_mean, z_log_var])

    encoder = keras.Model(e_input_layer, z, name="encoder")

    encoder.summary()
    encoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    d_input_layer = keras.Input(shape=(compressed_size,))
    layer = layers.Dense(conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3], activation="relu")(d_input_layer)
    layer = layers.Reshape(conv_output.shape[1:4])(layer)
    layer = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(layer)
    layer = Conv2DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(layer)

    decoder = keras.Model(d_input_layer, layer, name="decoder")

    decoder.summary()
    decoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    autoencoder = keras.Model(e_input_layer, decoder(encoder(e_input_layer)), name="auto-encoder")

    autoencoder.summary()
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return autoencoder, encoder, decoder
