import numpy as np
import tensorflow as tf
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose
from model import get_model
import os

from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"

directory = r'./data'

autoencoder, encoder, decoder = get_model()

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

#autoencoder.fit(train_data, train_data, epochs = 2, validation_data = (valid_data, valid_data), verbose = 1)
#autoencoder.save_weights("weights")

load_status = autoencoder.load_weights("weights")



d = data[0].reshape((28,28))
plt.imshow(d, cmap="gray")
lattentvector = encoder.predict(np.array([d]))
show_count = len(lattentvector) 
print(f"LATTENTVECTOR:{lattentvector}\n\n")

var = [-10, -1, 0, 1, 10]

for k in range(10):
    plt.figure(figsize=(14, 4))
    for i in range(show_count):
        for j in range(len(var)):
            tmp_latt = np.copy(lattentvector)
            tmp_latt[i] += var[j]
            print(tmp_latt)
            res = decoder.predict(tmp_latt).reshape((28,28))
            plt.subplot(show_count, 5, 1 + j + len(var)*i)
            plt.imshow(res, cmap="gray")
            plt.axis('off')
            plt.colorbar()
    plt.tight_layout()
    plt.show()


res = decoder.predict(lattentvector)[0].reshape((28,28))
plt.imshow(res, cmap="gray")
plt.show()
