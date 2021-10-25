import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose
from tensorflow.keras import optimizers

compressed_size = 8


class Sampling(layers.Layer):
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


input_layer = keras.Input(shape=(28, 28))

layer = Conv2D(16, 3, activation="relu", strides=2, padding="same")(input_layer)
conv_output = Conv2D(32, 3, activation="relu", strides=2, padding="same")(layer)
layer = Flatten()(conv_output)
layer = Dense(16, activation="relu")(layer)

z_mean = Dense(compressed_size, name="z_mean")(layer)
z_log_var = Dense(compressed_size, name="z_log_var")(layer)

z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(input_layer, layer, name="encoder")

encoder.summary()
encoder.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss="mse", metrics=["mae"])

input_layer = keras.Input(shape=(compressed_size,))
layer = layers.Dense(conv_output.shape[0] * conv_output.shape[1] * conv_output.shape[2], activation="relu")(input_layer)
layer = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(layer)
layer = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(layer)

decoder = keras.Model(input_layer, layer, name="decoder")

decoder.summary()
decoder.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss="mse", metrics=["mae"])
