import numpy as np
from matplotlib import pyplot as plt

# a)
n = np.arange(0, 100)
y = np.concatenate( (np.zeros(500), np.cos(2 * np.pi * 0.1 * n), np.zeros(300)) )


# b)
y_n = y + np.sqrt(0.5) * np.random.randn(y.size)


# c)

h =  np.cos(2 * np.pi * 0.1 * n)
detect = np.convolve(y_n, np.flip(h), mode="same")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
ax1.plot(y)
ax1.set_title("Noiseless signal")

ax2.plot(y_n)
ax2.set_title("Noisy signal")

ax3.plot(detect)
ax3.set_title("Detection Results with cos convolution")

h_f = np.exp(-2 * np.pi * 1j * 0.1 * n)
detect_freq = abs(np.convolve(y_n, np.flip(h_f), mode="same"))
ax4.plot(detect_freq)
ax4.set_title("Detection Results with exp convolution (abs)")

plt.show()