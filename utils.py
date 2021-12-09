import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def load_file(filename, max_data, directory="./data"):
    print(directory + '/' + filename)
    f = np.load(directory + '/' + filename, mmap_mode='r') / 255.0 # normalize to [0.0, 1.0] range
    if max_data != None:
        f = f[:min(len(f), max_data)]
    images = f.reshape((f.shape[0], 28, 28))
    categories = [os.path.splitext(os.path.basename(filename))[0].split("bitmap_")[1] for _ in range(len(f))]
    return images, categories

def load_files(directory="./data", max_data=100000):
    images = np.empty((0, 28, 28))
    categories = []
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            files.append(filename)
    for filename in files:
        imgs, cats = load_file(filename, max_data // len(files), directory = directory)
        images = np.append(images, imgs, axis=0)
        categories += cats
    print("Images loaded")
    return images, np.array(categories)


def make_a_gif_2(decoder, vectors, gif=False, name="dynamic_images"):
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


def make_a_gif(decoder, vector1, vector2, steps, gif=False, name="dynamic_images"):
    vectors = []
    vectors.append(vector1)
    inbetweens = np.linspace(vector1, vector2, steps, endpoint=False)
    for inb in inbetweens:
        vectors.append(inb)
    vectors.append(vector2)

    make_a_gif_2(vectors, gif, name)