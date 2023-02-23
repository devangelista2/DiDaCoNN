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

# Disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Load (test) data
data_path = './data/celeba_64/'
x_test = np.load(data_path + 'test.npy')

# Normalize
x_test = (x_test / 255.).astype(np.float32)

# Choose the ground-truth image.
idx = 20
x_true = x_test[idx:idx+1]
_, m, n, _ = x_true.shape

# Define the corruption model
kernel = iputils.get_gaussian_kernel(k=11, sigma=1.3)
K = operators.ConvolutionOperator(kernel, (m, n))

# Generate the corrupted image
y = K @ x_true
y = y.reshape((1, m, n, 1))

delta = 0.02
y_delta = y + delta * np.random.normal(0, 1, y.shape)

# Visualize
plt.imsave('./results/x_true.png', x_true[0, :, :, 0], cmap='gray')
plt.imsave('./results/y.png', y[0, :, :, 0], cmap='gray')
plt.imsave('./results/y_delta.png', y_delta[0, :, :, 0], cmap='gray')

# Naive inversion
compute_naive = False
if compute_naive:
    naive_solver = solvers.CGLS(K)
    x_naive = naive_solver(y_delta.flatten(), np.zeros_like(y_delta.flatten())).reshape((m, n))

    # Visualize Naive solution
    plt.imsave('./results/x_naive.png', x_naive, cmap='gray')

# Regularized inversion (Tikhonov)
compute_tik = False
if compute_tik:
    param_reg = 0.12
    A = operators.TikhonovOperator(K, operators.Identity(param_reg, (m, n)))
    
    tik_solver = stabilizers.Tik_CGLS_stabilizer(kernel, param_reg, k=200)
    x_tik = tik_solver(y_delta[0, :, :, 0]).reshape((m, n))

    # Visualize Naive solution
    plt.imsave('./results/x_tik.png', x_tik, cmap='gray')

# DiDeCoNN_random
compute_dideconn_random = False
if compute_dideconn_random:
    print("Computing Random solution.")
    # Get the trained model
    weights_filename = "diffusion_celeba_grayscale.hdf5"
    model = utils.get_trained_model(weights_filename, x_test)
    G = utils.generator_from_model(model)

    # Convert x_true to tf Tensor
    x_true = tf.convert_to_tensor(x_true)

    # Compute the solution
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    tf_K = utils.tf_K(k=11, sigma=1.3)
    z_didaconn, loss_vec = optimizers.optimize(G, tf_K, y_delta, optimizer, tf.random.normal((1, m, n, 1)), x_true, M=200)
    x_didaconn = G(z_didaconn)

    # Visualize Naive solution
    plt.imsave('./results/x_didaconn_random.png', x_didaconn[0, :, :, 0], cmap='gray')

    # Save the error
    np.save('./error_plots/didaconn_random_ssim.npy', loss_vec)

# DiDeCoNN_blurred
compute_didaconn_blurred = False
if compute_didaconn_blurred:
    print("Computing Blurred solution.")
    # Get the trained model
    weights_filename = "diffusion_celeba_grayscale.hdf5"
    model = utils.get_trained_model(weights_filename, x_test)
    G = utils.generator_from_model(model)

    # Convert x_true to tf Tensor
    x_true = tf.convert_to_tensor(x_true)

    # Compute the solution
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    tf_K = utils.tf_K(k=11, sigma=1.3)
    z_didaconn, loss_vec = optimizers.optimize(G, tf_K, y_delta, optimizer, tf.convert_to_tensor(y_delta * 0.02 / 0.95, tf.float32), x_true, M=200)
    x_didaconn = G(z_didaconn)

    # Visualize Naive solution
    plt.imsave('./results/x_didaconn_blurred.png', x_didaconn[0, :, :, 0], cmap='gray')

    # Save the error
    np.save('./error_plots/didaconn_blurred_ssim.npy', loss_vec)

# DiDeCoNN_true
compute_didaconn_true = False
if compute_didaconn_true:
    print("Computing True solution.")
    # Get the trained model
    weights_filename = "diffusion_celeba_grayscale.hdf5"
    model = utils.get_trained_model(weights_filename, x_test)
    G = utils.generator_from_model(model)

    # Convert x_true to tf Tensor
    x_true = tf.convert_to_tensor(x_true)

    # Compute the solution
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    tf_K = utils.tf_K(k=11, sigma=1.3)
    z_didaconn, loss_vec = optimizers.optimize(G, tf_K, y_delta, optimizer, tf.convert_to_tensor(x_true * 0.02 / 0.95, tf.float32), x_true, M=200)
    x_didaconn = G(z_didaconn)

    # Visualize Naive solution
    plt.imsave('./results/x_didaconn_true.png', x_didaconn[0, :, :, 0], cmap='gray')

    # Save the error
    np.save('./error_plots/didaconn_true_ssim.npy', loss_vec)

# With inversion
compute_didaconn_inversion = False
if compute_didaconn_inversion:
    print("Computing Starting point.")
    # Get the trained model
    weights_filename = "diffusion_celeba_grayscale.hdf5"
    model = utils.get_trained_model(weights_filename, x_test)
    G = utils.generator_from_model(model)

    # Get starting point
    import GD_model
    compute_starting_point = True
    if compute_starting_point:
        _, z0 = GD_model.diffusion_descent(model, x_true[0])
        print('Done!')

        # Save the starting point
        np.save('z0.npy', z0.numpy())

        # And visualize it
        plt.imsave('x0.png', G(z0)[0, :, :, 0], cmap='gray')
    else:
        z0 = np.load('z0.npy')

    # Compute the solution
    print('Solving the deblurring problem')

    # Convert x_true to tf Tensor
    x_true = tf.convert_to_tensor(x_true)

    # Compute the solution
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    tf_K = utils.tf_K(k=11, sigma=1.3)
    z_didaconn, loss_vec = optimizers.optimize(G, tf_K, y_delta, optimizer, z0, x_true, M=200)
    x_didaconn = G(z_didaconn)

    # Visualize Naive solution
    plt.imsave('./results/x_didaconn_inversion.png', x_didaconn[0, :, :, 0], cmap='gray')

    # Save the error
    np.save('./error_plots/didaconn_inversion_ssim.npy', loss_vec)

# With blurred inversion
compute_didaconn_blurred_inversion = True
if compute_didaconn_blurred_inversion:
    print("Computing Starting point.")
    # Get the trained model
    weights_filename = "diffusion_celeba_grayscale.hdf5"
    model = utils.get_trained_model(weights_filename, x_test)
    G = utils.generator_from_model(model)

    # Get starting point
    import GD_model
    compute_starting_point = False
    if compute_starting_point:
        _, z0 = GD_model.diffusion_descent(model, y_delta[0])
        print('Done!')

        # Save the starting point
        np.save('z0_blurred.npy', z0.numpy())

        # And visualize it
        plt.imsave('x0.png', G(z0)[0, :, :, 0], cmap='gray')
    else:
        z0 = np.load('z0_blurred.npy')

    # Compute the solution
    print('Solving the deblurring problem')

    # Convert x_true to tf Tensor
    x_true = tf.convert_to_tensor(x_true)

    # Compute the solution
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-2)
    tf_K = utils.tf_K(k=11, sigma=1.3)
    z_didaconn, loss_vec = optimizers.optimize(G, tf_K, y_delta, optimizer, z0, x_true, M=200)
    x_didaconn = G(z_didaconn)

    # Visualize Naive solution
    plt.imsave('./results/x_didaconn_blurred_inversion.png', x_didaconn[0, :, :, 0], cmap='gray')

    # Save the error
    np.save('./error_plots/didaconn_blurred_inversion_ssim.npy', loss_vec)

