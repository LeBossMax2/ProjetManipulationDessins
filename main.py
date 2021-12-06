import numpy as np
import tensorflow as tf
import sklearn.model_selection
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping

import os

from matplotlib import pyplot as plt

compressed_size = 64
lambda_loss = 1e-5
directory = r'./data'

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


# learning_rate=0.002
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

def load_files():
    files = np.empty((0, 28, 28, 1))
    categories = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy') :
            print(directory + '/' + filename)
            f = np.load(directory + '/' + filename, mmap_mode='r') / 255.0 # normalize to [0.0, 1.0] range
            files = np.append(files, f.reshape((f.shape[0], 28, 28, 1)), axis=0)
            categories += [ os.path.basename(filename) for _ in range(len(f))]
    print("Files loaded")
    return files, np.array(categories)

data, categories = load_files()

print(f"Shape of data: {data.shape}")

train_data, valid_data, train_cat, valid_cat = sklearn.model_selection.train_test_split(data, categories, test_size=0.33, shuffle=True)

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

hist = autoencoder.fit(train_data, train_data, batch_size = 128, epochs = 20, validation_data = (valid_data, valid_data), verbose = 2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss", min_delta=0)])


autoencoder.save_weights("weights")
#load_status = autoencoder.load_weights("weights")

def plot_label_clusters(encoder, x, y):
    # display a 2D plot of the digit classes in the latent space

    categories = np.unique(y)
    vals = [x[y == cat] for cat in categories]

    plt.figure(figsize=(18, 10))
    for zi in range(0, compressed_size, 2):
        plt.subplot(4, compressed_size // 8, zi // 2 + 1)
        for cat, val in zip(categories, vals):
            z = encoder.predict(val)
            plt.scatter(z[:, zi], z[:, zi + 1], label=cat, alpha=.2, s=3**2)
        plt.xlabel("z[" + str(zi) + "]")
        plt.ylabel("z[" + str(zi + 1) + "]")
    plt.show()

plot_label_clusters(encoder, valid_data, valid_cat)

# Show prediction examples
show_count = 6
for k in range(100):
    plt.figure(figsize=(14, 4))
    for i in range(show_count):
        d = valid_data[i + k * show_count].reshape((28,28))
        plt.subplot(2, show_count, 1 + i)
        plt.imshow(d, cmap="gray")
        plt.colorbar()
        plt.axis('off')
        res = autoencoder.predict(np.array([d]))[0].reshape((28,28))
        plt.subplot(2, show_count, 1 + i + show_count)
        plt.imshow(res, cmap="gray")
        plt.axis('off')
        plt.colorbar()
    plt.tight_layout()
    plt.show()
