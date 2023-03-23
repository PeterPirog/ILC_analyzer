import numpy as np
import matplotlib.pyplot as plt
def plot_x_with_uncertainty(x, sigma, mu, unc_alea, k=1):
    plt.errorbar(range(len(x)), x, yerr=sigma, fmt='o')
    plt.xlabel('Laboratory index')
    plt.ylabel('Value of mean with standard deviation')
    plt.title(f'Values of laboratories means with uncertainty k={k}')
    plt.axhline(y=mu, color='r', linestyle='-')  # Horizontal line at mu
    plt.axhline(y=mu + unc_alea*k, color='g', linestyle='--')  # Dashed line at mu+unc_alea
    plt.axhline(y=mu - unc_alea*k, color='g', linestyle='--')  # Dashed line at mu-unc_alea
    plt.show()