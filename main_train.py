import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import model_VAE
from utils import load_files

from matplotlib import pyplot as plt

autoencoder, encoder, decoder = model_VAE.get_model()

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

hist = autoencoder.fit(train_data, train_data, batch_size = 128, epochs = 50, validation_data = (valid_data, valid_data), verbose = 2, callbacks=[EarlyStopping(patience=2, monitor="val_loss", min_delta=0)])


autoencoder.save_weights("weights")
#load_status = autoencoder.load_weights("weights")

# Show loss history
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

def plot_label_clusters(encoder, x, y):
    # display a 2D plot of the digit classes in the latent space

    categories = np.unique(y)
    vals = [x[y == cat] for cat in categories]

    plt.figure(figsize=(18, 10))
    for zi in range(0, model_VAE.compressed_size, 2):
        plt.subplot(4, model_VAE.compressed_size // 8, zi // 2 + 1)
        for cat, val in zip(categories, vals):
            z = encoder.predict(val)
            plt.scatter(z[:, zi], z[:, zi + 1], label=cat, alpha=.1, s=3**2)
        plt.xlabel("z[" + str(zi) + "]")
        plt.ylabel("z[" + str(zi + 1) + "]")
    plt.tight_layout()
    plt.show()

plot_label_clusters(encoder, valid_data, valid_cat)

# Show prediction examples
show_count = 6
for k in range(2):
    plt.figure(figsize=(14, 4))
    for i in range(show_count):
        d = valid_data[i + k * show_count]
        plt.subplot(2, show_count, 1 + i)
        plt.imshow(d, cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
        res = autoencoder.predict(np.array([d]))[0]
        plt.subplot(2, show_count, 1 + i + show_count)
        plt.imshow(res, cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
