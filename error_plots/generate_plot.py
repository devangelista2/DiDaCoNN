import numpy as np
import matplotlib.pyplot as plt

random = np.load('./error_plots/didaconn_random_ssim.npy')
blurred = np.load('./error_plots/didaconn_blurred_ssim.npy')
true = np.load('./error_plots/didaconn_true_ssim.npy')
inversion = np.load('./error_plots/didaconn_inversion_ssim.npy')
blurred_inversion = np.load('./error_plots/didaconn_blurred_inversion_ssim.npy')

# Plot
plt.figure()
plt.plot(np.linspace(1, len(random), len(random)), random)
plt.plot(np.linspace(1, len(blurred), len(blurred)), blurred)
plt.plot(np.linspace(1, len(true), len(true)), true)
plt.plot(np.linspace(1, len(inversion), len(inversion)), inversion)
plt.plot(np.linspace(1, len(blurred_inversion), len(blurred_inversion)), blurred_inversion)
plt.grid()
plt.legend(['Random', 'Blurred', 'True', 'Inversion', 'Blurred Inversion'])
plt.xlabel('iteration')
plt.ylabel('SSIM')
plt.savefig('./error_plots/SSIM_plot.png', dpi=500)