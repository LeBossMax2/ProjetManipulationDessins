import numpy as np
import sklearn.model_selection
from utils import load_files, make_a_gif
from model_VAE import get_model, train_and_save_weights
import os

from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="1"

directory = r'./data'

autoencoder, encoder, decoder = get_model()

def plot_loss_acc(history):
    """Plot training and (optionally) validation loss and accuracy"""

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, '.--', label='Training loss')
    final_loss = loss[-1]
    title = 'Training loss: {:.4f}'.format(final_loss)
    plt.ylabel('Loss')
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'o-', label='Validation loss')
        final_val_loss = val_loss[-1]
        title += ', Validation loss: {:.4f}'.format(final_val_loss)
    plt.title(title)
    plt.legend()

    acc = history.history['accuracy']

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, '.--', label='Training acc')
    final_acc = acc[-1]
    title = 'Training accuracy: {:.2f}%'.format(final_acc * 100)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if 'val_accuracy' in history.history:
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, val_acc, 'o-', label='Validation acc')
        final_val_acc = val_acc[-1]
        title += ', Validation accuracy: {:.2f}%'.format(final_val_acc * 100)
    plt.title(title)
    plt.legend()

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
            make_a_gif(decoder, ultimas[ulti1], ultimas[ulti2], nb_steps, True)





data, categories = load_files(directory)

print(f"Shape of data: {data.shape}")

train_data, valid_data, train_cat, valid_cat = sklearn.model_selection.train_test_split(data, categories, test_size=0.33)

print("Data ready")

history = train_and_save_weights(autoencoder, train_data, valid_data)
plot_loss_acc(history)

load_status = autoencoder.load_weights("weights")

#ultimas, categories = print_mean(valid_data, valid_cat)
#transit(ultimas, categories, 7)

#test_each_dimension(encoder, decoder, valid_data, var = [-4, -3, -2, -1, 0, 1, 2, 3, 4])

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
