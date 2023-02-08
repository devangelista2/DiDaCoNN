import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import layers

import os

import DDIM

# Load data
data_path = './data/celeba_64/'

x_train = np.load(data_path + 'train.npy')
x_test = np.load(data_path + 'test.npy')

# Normalize
x_train = (x_train / 255.).astype(np.float32)
x_test = (x_test / 255.).astype(np.float32)

# Model parameters
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
image_size = 64
embedding_dims = 64
embedding_max_frequency = 1000.0
depths = [48, 96, 192, 384]
block_depth = 2

### Training parameters
batch_size = 128
learning_rate = 0.0001
weight_decay = 1e-4

num_epochs = 50

# Get the model
model = DDIM.DDIM(image_size, batch_size, depths, block_depth, max_signal_rate,
                  min_signal_rate, embedding_dims, embedding_max_frequency)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.mean_absolute_error,
)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(x_test)

### Run training and plot generated images periodically
weights_filename = "diffusion_celeba_grayscale.hdf5"

model.fit(
    x_train,
    epochs=num_epochs,
    batch_size=batch_size
)
model.save_weights("weights/" + weights_filename)