import numpy as np
import matplotlib.pyplot as plt
import math

def main(samples=100):
    # Form a sinusoidal signal
    N = 160
    n = np.arange(N)
    f0 = 0.06752728319488948


    # Add noise to the signal
    sigmaSq = 0 # 1.2
    phi = 0.6090665392794814
    A = 0.6669548209299414

    x0 = A * np.cos(2 * np.pi * f0 * n+phi)
    x = x0 + sigmaSq * np.random.randn(x0.size)
    # Estimation parameters
    A_hat = A*10
    phi_hat = phi*-2
    fRange = np.linspace(0, 0.5, samples)

    # a)


    SSE = np.empty(fRange.size)
    ML = np.empty(fRange.size)
    for i,frequency in enumerate(fRange):
        x0 = A_hat * np.cos(2 * np.pi * frequency * n+phi_hat)

        SSE[i] = np.sum((x0 - x)**2)

        sigma = np.std(x)
        ML[i] = (np.prod((1/np.sqrt(2*np.pi*sigma**2)) * np.exp( (-(1/(2*sigma**2)) *(x0 - x)**2 ) )))

    if np.argmin(SSE) == np.argmax(ML):
        estim_f = fRange[np.argmax(ML)]
        print("Same estim_f for ML and SSE, estimated frequency {}".format(estim_f))
    else:
        estim_f = fRange[np.argmin(ML)]
        print("Not the same estim_f for ML and SSE, using ML, estimated frequency {}".format(estim_f))


    print("Actual frequency {}".format(f0))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ax1.plot(fRange, SSE)
    ax1.set_title("Squared error")

    ax2.plot(fRange, ML)
    ax2.set_title("Likelihood")

    x0 = A * np.cos(2 * np.pi * f0 * n+phi)
    x = x0 + sigmaSq * np.random.randn(x0.size)

    ax3.plot(n, x0, n[0:-1:1], x[0:-1:1], "g.")
    ax3.set_title("Signal and noisy samples for sigmaSq = {:.5f}".format(sigmaSq))
    ax3.legend(["Acos(2pif0+phi)", "Noisy samples"])

    estim_x = A * np.cos(2 * np.pi * estim_f * n+phi)

    ax4.plot(n, x0, "b" ,n, estim_x, "r--")
    ax4.set_title("True f0 = {:.5f} (blue) and estimated  f0 = {:.5f} (red)".format(f0, estim_f))


    plt.show()

main()