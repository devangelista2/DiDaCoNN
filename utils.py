from settings import *
import numpy as np

from tensorflow import keras
import tensorflow as tf
from IPPy import utils as iputils
import DDIM

def get_trained_model(weights_filename, x_test):
    # Get the model
    model = DDIM.DDIM(model_params['image_size'], training_params['batch_size'], model_params['depths'], 
                    model_params['block_depth'], model_params['max_signal_rate'], model_params['min_signal_rate'], 
                    model_params['embedding_dims'], model_params['embedding_max_frequency'])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_params['learning_rate']),
        loss=keras.losses.mean_absolute_error,
    )

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(x_test)

    # Load the weights
    model.load_weights("weights/" + weights_filename)
    return model

# Define the generator
def generator_from_model(model):
    def G(z):
        x_gen = model.reverse_diffusion(z, diffusion_steps=10)
        return model.denormalize(x_gen)
    return G

def tf_K(k, sigma):
    def K(x):
        x = tf.cast(x, tf.float32)
        kernel = iputils.get_gaussian_kernel(k, sigma)
        kernel = tf.convert_to_tensor(kernel)
        kernel = tf.reshape(kernel, kernel.shape + (1, 1))
        kernel = tf.cast(kernel, tf.float32)
        return tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
    return K