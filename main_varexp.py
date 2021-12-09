import numpy as np
import tensorflow as tf
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose
from model_AE import get_model
from utils import load_files
import os

from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"

directory = r'./data'

autoencoder, encoder, decoder = get_model()

def print_mean(data_):
    plt.figure(figsize=(14, 4))
    ultimate_mean_image = np.mean(data_, axis=0)
    plt.subplot(1, 2, 1)
    plt.imshow(ultimate_mean_image, cmap="gray")
    mean_data = np.mean(encoder.predict(data_), axis=0)
    res = decoder.predict(np.array([mean_data])).reshape((28,28))
    plt.subplot(1, 2, 2)
    plt.imshow(res, cmap="gray")
    plt.show()

def test_each_dimension(encoder, decoder, data, var):
    for k in range(10):
        d = data[k].reshape((28,28))
        plt.imshow(d, cmap="gray")
        lattentvector = encoder.predict(np.array([d]))
        show_count = len(lattentvector[0]) 
        print(f"LATTENTVECTOR:{lattentvector}\n\n")

        plt.figure(figsize=(14, 4))
        for i in range(show_count):
            for j in range(len(var)):
                tmp_latt = np.copy(lattentvector)
                tmp_latt[0][i] += var[j]
                res = decoder.predict(tmp_latt).reshape((28,28))
                plt.subplot(len(var), show_count, 1 + j*show_count + i)
                plt.imshow(res, cmap="gray")
                plt.axis('off')
                #plt.colorbar()
        plt.tight_layout()
        plt.show()

data = load_files(directory)

print(f"Shape of data: {data.shape}")

train_data, valid_data = sklearn.model_selection.train_test_split(data, test_size=0.33)

print("Data ready")

#autoencoder.fit(train_data, train_data, epochs = 2, validation_data = (valid_data, valid_data), verbose = 1)
#autoencoder.save_weights("weights")

load_status = autoencoder.load_weights("weights")

print_mean(valid_data)

test_each_dimension(encoder, decoder, data, var = [-4, -3, -2, -1, 0, 1, 2, 3, 4])