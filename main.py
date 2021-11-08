import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose
from tensorflow.keras import optimizers

import os

from matplotlib import pyplot as plt

compressed_size = 8
directory = r'data'

class Sampling(layers.Layer):
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


input_layer = keras.Input(shape=(28, 28, 1))

layer = Conv2D(16, 3, activation="relu", strides=2, padding="same")(input_layer)
conv_output = Conv2D(32, 3, activation="relu", strides=2, padding="same")(layer)
layer = Flatten()(conv_output)
z = Dense(compressed_size, activation="relu")(layer)

encoder = keras.Model(input_layer, z, name="encoder")

encoder.summary()
encoder.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss="mse", metrics=["mae"])

input_layer = keras.Input(shape=(compressed_size,))
layer = layers.Dense(conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3], activation="relu")(input_layer)
layer = layers.Reshape(conv_output.shape[1:4])(layer)
layer = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(layer)
layer = Conv2DTranspose(1, 3, activation="relu", strides=2, padding="same")(layer)

decoder = keras.Model(input_layer, layer, name="decoder")

decoder.summary()
decoder.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss="mse", metrics=["mae"])

def load_files():
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy') :
            print(directory + '/' + filename)
            files = np.append(files, np.load(directory + '/' + filename, mmap_mode='r'))
    print("Files loaded")
    return np.random.shuffle(files)

files = load_files()
