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

load_status = autoencoder.load_weights("weights")

show_count = 6
d = valid_data[0]
plt.imshow(d, cmap="gray")
lattentvector = encoder.predict(np.array([d]))[0]
print(f"LATTENTVECTOR:{lattentvector}\n\n")
res = decoder.predict(lattentvector)[0]
plt.imshow(res, cmap="gray")
plt.show()
