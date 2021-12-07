import numpy as np
import os

def load_file(filename, max_data, directory="./data"):
    print(directory + '/' + filename)
    f = np.load(directory + '/' + filename, mmap_mode='r') / 255.0 # normalize to [0.0, 1.0] range
    if max_data != None:
        f = f[:min(len(f), max_data)]
    images = f.reshape((f.shape[0], 28, 28))
    categories = [os.path.splitext(os.path.basename(filename))[0] for _ in range(len(f))]
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
