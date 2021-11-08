import numpy as np
import tensorflow as tf
import sklearn.model_selection
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose

import os

from matplotlib import pyplot as plt

compressed_size = 8
directory = r'../data'

class Sampling(layers.Layer):
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


e_input_layer = keras.Input(shape=(28, 28, 1))

layer = Conv2D(16, 3, activation="relu", strides=2, padding="same")(e_input_layer)
conv_output = Conv2D(32, 3, activation="relu", strides=2, padding="same")(layer)
layer = Flatten()(conv_output)
z = Dense(compressed_size, activation="relu")(layer)

encoder = keras.Model(e_input_layer, z, name="encoder")

encoder.summary()
encoder.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss="mse", metrics=["mae"])

d_input_layer = keras.Input(shape=(compressed_size,))
layer = layers.Dense(conv_output.shape[1] * conv_output.shape[2] * conv_output.shape[3], activation="relu")(d_input_layer)
layer = layers.Reshape(conv_output.shape[1:4])(layer)
layer = Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(layer)
layer = Conv2DTranspose(1, 3, activation="relu", strides=2, padding="same")(layer)

decoder = keras.Model(d_input_layer, layer, name="decoder")

decoder.summary()
decoder.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss="mse", metrics=["mae"])

autoencoder = keras.Model(e_input_layer, decoder(z), name="auto-encoder")

autoencoder.summary()
autoencoder.compile(optimizer=optimizers.SGD(learning_rate=0.1), loss="mse", metrics=["mae"])

def load_files():
    files = np.empty((0, 28, 28, 1))
    for filename in os.listdir(directory):
        if filename.endswith('.npy') :
            print(directory + '/' + filename)
            f = np.load(directory + '/' + filename, mmap_mode='r')
            files = np.append(files, f.reshape((f.shape[0], 28, 28, 1)), axis=0)
    print("Files loaded")
    np.random.shuffle(files)
    return files

data = load_files()

print(f"Shape of data: {data.shape}")

train_data, valid_data = sklearn.model_selection.train_test_split(data, test_size=0.33)

print("Data ready")

autoencoder.fit(train_data, train_data, epochs = 4, validation_data = (valid_data, valid_data), verbose = 2)
