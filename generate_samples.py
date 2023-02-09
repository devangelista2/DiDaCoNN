import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from settings import *
import utils

# Load (test) data
data_path = './data/celeba_64/'
x_test = np.load(data_path + 'test.npy')

# Normalize
x_test = (x_test / 255.).astype(np.float32)

# Get the trained model
weights_filename = "diffusion_celeba_grayscale.hdf5"
model = utils.get_trained_model(weights_filename, x_test)

# Generate one image
z = tf.random.normal((3, 64, 64, 1))
x_gen = model.reverse_diffusion(z, 10)

# Save it
plt.figure(figsize=(18, 6))
for i in range(len(x_gen)):
    plt.subplot(1, len(x_gen), i+1)
    plt.imshow(x_gen[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('./results/generated_sample.png', dpi=500)
