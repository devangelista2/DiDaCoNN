import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras import layers

import os

import DDIM
from settings import *

# Load data
data_path = './data/celeba_64/'

x_train = np.load(data_path + 'train.npy')
x_test = np.load(data_path + 'test.npy')

# Normalize
x_train = (x_train / 255.).astype(np.float32)
x_test = (x_test / 255.).astype(np.float32)

# To make the training faster, consider just a subset of the training set
N = 50_000
idx = np.random.choice(np.arange(N), N)

x_train = x_train[idx]

# Get the model
model = DDIM.DDIM(model_params['image_size'], model_params['batch_size'], model_params['depths'], 
                  model_params['block_depth'], model_params['max_signal_rate'], model_params['min_signal_rate'], 
                  model_params['embedding_dims'], model_params['embedding_max_frequency'])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=training_params['learning_rate']),
    loss=keras.losses.mean_absolute_error,
)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(x_test)

### Run training and plot generated images periodically
weights_filename = "diffusion_celeba_grayscale.hdf5"

model.fit(
    x_train,
    epochs=training_params['num_epochs'],
    batch_size=training_params['batch_size']
)
model.save_weights("weights/" + weights_filename)