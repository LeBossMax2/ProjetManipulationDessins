import numpy as np
import os


def load_files(directory="./data"):
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
