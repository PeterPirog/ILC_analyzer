import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ILC_analyzer.nigtools import (
    estimate_nig_params,
    normal_inverse_gamma,
)

if __name__ == "__main__":
    path_data = '../data/ILC_data.xlsx'

    # Load data from excel file
    df = pd.read_excel(path_data)
    x_i = np.array(df['Avg'].values)
    sigma_i = np.array(df['std'].values)

    # Calculate parameters of normal-inverse-gamma distribution
    mu, lambda_, alpha, beta = estimate_nig_params(x_i, sigma_i)
    print(f"Estimated parameters of NIG distribution: mu={mu:.4f}, lambda={lambda_:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")


    # Ustawienie zakresów x i sigma
    x_min, x_max = 0.5, 3.0
    sigma_min, sigma_max = 1e-10, 1.0

    # Obliczenie wartości funkcji w siatce punktów (x, sigma)
    x = np.linspace(x_min, x_max, 100)
    sigma = np.linspace(sigma_min, sigma_max, 100)
    X, Y = np.meshgrid(x, sigma)
    Z = normal_inverse_gamma(X, Y, mu, lambda_, alpha, beta)

    # Narysowanie wykresu 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('sigma')
    ax.set_zlabel('f(x, sigma)')
    ax.set_title('Wykres 3D')

    # Dodanie punktów na wykresie 2D

    # Narysowanie wykresu 2D z kolorem
    ax2 = fig.add_subplot(122)
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('sigma')
    plt.title('Wykres 2D z kolorem')
    plt.colorbar()

    # Dodanie konturu oznaczającego obszar prawdopodobieństwa
    ax2.contour(X, Y, Z, levels=[1.6], colors='red')

    # Dodanie punktów na wykresie 2D
    plt.scatter(x_i, sigma_i, marker='+', color='black')
    plt.tight_layout()
    plt.show()

