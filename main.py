import numpy as np
import tensorflow as tf
import sklearn.model_selection
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose

import os

from matplotlib import pyplot as plt

compressed_size = 32
directory = r'./data'

class VariationalLayer(layers.Layer):
    
    def call(self, inputs):
        z_mean, z_log_var = inputs

        # KL divergence loss
        kl_batch = K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        
        self.add_loss(-0.5 * K.mean(kl_batch) / (28 * 28), inputs = inputs)

        # Sampling reparameterization
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


optimizer = optimizers.Adam(learning_rate=0.002)
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

def load_files():
    files = np.empty((0, 28, 28, 1))
    for filename in os.listdir(directory):
        if filename.endswith('.npy') :
            print(directory + '/' + filename)
            f = np.load(directory + '/' + filename, mmap_mode='r') / 255.0 # normalize to [0.0, 1.0] range
            files = np.append(files, f.reshape((f.shape[0], 28, 28, 1)), axis=0)
    print("Files loaded")
    np.random.shuffle(files)
    return files

data = load_files()

print(f"Shape of data: {data.shape}")

train_data, valid_data = sklearn.model_selection.train_test_split(data, test_size=0.33)

print("Data ready")
'''
# Show prediction examples
show_count = 4
fig, axes = plt.subplots(nrows=show_count, ncols=2)

for i in range(0,8,2)  :
    im = axes.flat[i].imshow(data[i], cmap="gray")
    axes.flat[i].axis('off')
    res = autoencoder.predict(np.array([data[i]]))[0]
    axes.flat[i+1].imshow(res, cmap="gray")
    axes.flat[i+1].axis('off')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()
'''

hist = autoencoder.fit(train_data, train_data, batch_size = 128, epochs = 4, validation_data = (valid_data, valid_data), verbose = 1)


autoencoder.save_weights("weights")
#load_status = autoencoder.load_weights("weights")

def plot_label_clusters(encoder, x, y):
    # display a 2D plot of the digit classes in the latent space
    z = encoder.predict(x)
    plt.figure(figsize=(12, 10))
    plt.scatter(z[:, 0], z[:, 1], c=y, alpha=.4, s=3**2)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

plot_label_clusters(encoder, valid_data, None)

# Show prediction examples
show_count = 6
for k in range(100):
    plt.figure(figsize=(14, 4))
    for i in range(show_count):
        d = valid_data[i + k * show_count]
        plt.subplot(2, show_count, 1 + i)
        plt.imshow(d, cmap="gray")
        plt.colorbar()
        plt.axis('off')
        res = autoencoder.predict(np.array([d]))[0]
        plt.subplot(2, show_count, 1 + i + show_count)
        plt.imshow(res, cmap="gray")
        plt.axis('off')
        plt.colorbar()
    plt.tight_layout()
    plt.show()
