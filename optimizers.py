import matplotlib.pyplot as plt
import numpy as np
import math

import os
import progressbar

from IPPy import utils

from tensorflow import keras
from keras import layers
import tensorflow as tf

def optimize(G, K, y_delta, optimizer, z, x_true, M=150):
    # Change z to a variable.
    z = tf.Variable(z)

    loss_vec = np.zeros((M+1, ))

    # Initialize progressbar
    widgets = [' [',
         progressbar.Timer(format= 'Time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.Percentage(), ') ',
          ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=M+1).start()

    # Compute the starting iterate and the corresponding loss
    x_k = G(z)
    loss_vec[0] = tf.image.ssim(x_k, x_true, 1)

    # Run the Gradient Descent
    for k in range(1,M+1):
        with tf.GradientTape() as tape:
            tape.watch(z)
            x_k = G(z)
            d = tf.reduce_mean(tf.square(K(x_k) - y_delta))
        gradient = tape.gradient(d, z)
        loss_vec[k] = tf.image.ssim(x_k, x_true, 1)

        # Update z
        optimizer.apply_gradients(zip([gradient], [z]))

        # Update progressbar
        bar.update(k)

    return z, loss_vec