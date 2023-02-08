import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage.color import rgb2gray
# Path for the data
path = './data/celeba_64/train.npy'
path_to = './data/celeba_64/train2.npy'

# Load the dataset
data_rgb = np.load(path)


# Convert it to Gray-scale
data_gray = 0.2125 * data_rgb[:, :, :, 0] + 0.7154 * data_rgb[:, :, :, 1] + 0.0721 * data_rgb[:, :, :, 2]
data_gray = data_gray.astype(np.uint8)
data_gray = np.expand_dims(data_gray, -1)

np.save(path_to, data_gray)