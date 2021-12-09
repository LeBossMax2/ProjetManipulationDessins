import numpy as np
import sklearn.model_selection
from utils import load_files
from model_VAE import get_model
import os

from matplotlib import pyplot as plt
import matplotlib.animation as animation

os.environ["CUDA_VISIBLE_DEVICES"]="1"

directory = r'./data'

autoencoder, encoder, decoder = get_model()

def print_mean(data_, categories_):
    categories_unique = np.unique(categories_)
    data_categorised = [data_[categories_ == cat] for cat in categories_unique]
    ultimas = []
    for cat, dat in zip(categories_unique, data_categorised):
        plt.figure(figsize=(14, 4))
        ultimate_mean_image = np.mean(dat, axis=0)
        plt.subplot(1, 3, 1)
        plt.imshow(ultimate_mean_image, cmap="gray")
        plt.title("Mean image of all " + cat)
        lattent_vectors = encoder.predict(dat)
        mean_data = np.mean(lattent_vectors, axis=0)
        best_data_index = np.argmin([np.dot(lat - mean_data, lat - mean_data) for lat in lattent_vectors], axis=0)
        ultimas.append(lattent_vectors[best_data_index])
        res = decoder.predict(np.array([lattent_vectors[best_data_index]]))[0]
        plt.subplot(1, 3, 2)
        plt.imshow(res, cmap="gray")
        plt.title("Ultimate " + cat)
        plt.subplot(1, 3, 3)
        plt.imshow(dat[best_data_index], cmap="gray")
        plt.title("Base image for Ultimate " + cat)
        plt.show()
    return ultimas, categories_unique


def test_each_dimension(encoder, decoder, data, var):
    for k in range(10):
        d = data[k]
        plt.imshow(d, cmap="gray")
        lattentvector = encoder.predict(np.array([d]))
        show_count = len(lattentvector[0]) 
        print(f"LATTENTVECTOR:{lattentvector}\n\n")

        plt.figure(figsize=(14, 4))
        for i in range(show_count):
            for j in range(len(var)):
                tmp_latt = np.copy(lattentvector)
                tmp_latt[0][i] += var[j]
                res = decoder.predict(tmp_latt)[0]
                plt.subplot(len(var), show_count, 1 + j*show_count + i)
                plt.imshow(res, cmap="gray")
                plt.axis('off')
                #plt.colorbar()
        plt.tight_layout()
        plt.show()


def transit(ultimas, categories, nb_steps):
    cols = nb_steps + 2
    for ulti1 in range(len(ultimas)-1):
        for ulti2 in range(ulti1+1, len(ultimas)):
            plt.figure(figsize=(14, 4))
            plt.subplot(1, cols, 1)
            plt.imshow(decoder.predict(np.array([ultimas[ulti1]]))[0], cmap='gray')
            plt.title("Ultima " + categories[ulti1])
            plt.subplot(1, cols, cols)
            plt.imshow(decoder.predict(np.array([ultimas[ulti2]]))[0], cmap='gray')
            plt.title("Ultima " + categories[ulti2])
            inbetweens = np.linspace(ultimas[ulti1], ultimas[ulti2], nb_steps, endpoint=False)
            ctr = 0
            for inb in inbetweens:
                plt.subplot(1, cols, 2 + ctr)
                ctr += 1
                plt.imshow(decoder.predict(np.array([inb]))[0], cmap='gray')
            plt.tight_layout()
            plt.show()
            make_a_gif(ultimas[ulti1], ultimas[ulti2], nb_steps)


def make_a_gif_2(vectors, gif=False, name="dynamic_images"):
    fig = plt.figure()
    plt.axis('off')
    plt.title(name)
    ims = []
    for v in vectors:
        ims.append([plt.imshow(decoder.predict(np.array([v]))[0], cmap='gray', animated=True)])
    ani = animation.ArtistAnimation(fig, ims)
    if gif:
        ani.save(name + ".gif")
    plt.tight_layout()
    plt.show()


def make_a_gif(vector1, vector2, steps, gif=False, name="dynamic_images"):
    vectors = []
    vectors.append(vector1)
    inbetweens = np.linspace(vector1, vector2, steps, endpoint=False)
    for inb in inbetweens:
        vectors.append(inb)
    vectors.append(vector2)

    make_a_gif_2(vectors, gif, name)


data, categories = load_files(directory)

print(f"Shape of data: {data.shape}")

train_data, valid_data, train_cat, valid_cat = sklearn.model_selection.train_test_split(data, categories, test_size=0.33)

print("Data ready")

#autoencoder.fit(train_data, train_data, epochs = 3, validation_data = (valid_data, valid_data), verbose = 1)
#autoencoder.save_weights("weights")

load_status = autoencoder.load_weights("weights")

ultimas, categories = print_mean(valid_data, valid_cat)
transit(ultimas, categories, 7)

test_each_dimension(encoder, decoder, valid_data, var = [-4, -3, -2, -1, 0, 1, 2, 3, 4])