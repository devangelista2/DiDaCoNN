import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import progressbar
import os

from settings import *
import utils
from IPPy import utils as iputils
from IPPy import operators, solvers, stabilizers, metrics
import optimizers
import GD_model

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load (test) data
data_path = './data/celeba_64/'
x_test = np.load(data_path + 'test.npy')

# Normalize
x_test = (x_test / 255.).astype(np.float32)

# Get the trained model
weights_filename = "diffusion_celeba_grayscale.hdf5"
model = utils.get_trained_model(weights_filename, x_test)
G = utils.generator_from_model(model)

# Define the image x s.t. we search z with G(z) = x
x = x_test[20]
_, z = GD_model.diffusion_descent(model, x)

# And visualize it
plt.imsave('./test_range_generator/x_true.png', x[:, :, 0], cmap='gray')
plt.imsave('./test_range_generator/x_approx.png', G(z)[0, :, :, 0], cmap='gray')