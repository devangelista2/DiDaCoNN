import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from settings import *
import utils
from IPPy import utils as iputils
from IPPy import operators

# Load (test) data
data_path = './data/celeba_64/'
x_test = np.load(data_path + 'test.npy')

# Normalize
x_test = (x_test / 255.).astype(np.float32)

# Choose the ground-truth image.
idx = 20
x_true = x_test[idx:idx+1]
_, m, n, _ = x_true.shape

# Get the trained model
weights_filename = "diffusion_celeba_grayscale.hdf5"
model = utils.get_trained_model(weights_filename, x_test)