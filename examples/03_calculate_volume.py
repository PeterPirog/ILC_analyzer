import numpy as np
import pandas as pd
from ILC_analyzer.nigtools import (
    estimate_nig_params,
    aleatoric_uncertainy_from_nig,
    epistemic_uncertainy_from_nig,
)
from ILC_analyzer.plotting import plot_x_with_uncertainty

if __name__ == "__main__":
    path_data = '../data/ILC_data.xlsx'

    # Load data from excel file
    df = pd.read_excel(path_data)
    x_i = np.array(df['Avg'].values)
    sigma_i = np.array(df['std'].values)

    # Calculate parameters of normal-inverse-gamma distribution
    mu, lambda_, alpha, beta = estimate_nig_params(x_i, sigma_i)
    print(f"Estimated parameters of NIG distribution: mu={mu:.4f}, lambda={lambda_:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")

    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    def normal_inverse_gamma(x, y, mu, lambda_, alpha, beta):
        inv_gamma = stats.invgamma(alpha, scale=beta)
        normal = stats.norm(mu, np.sqrt(y / lambda_))
        return normal.pdf(x) * inv_gamma.pdf(y)

    def find_contour_points(f, mu, lambda_, alpha, beta, value, x_range, y_range, num_points=1000):
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x_values, y_values)
        Z = f(X, Y, mu, lambda_, alpha, beta)

        contour = plt.contour(X, Y, Z, levels=[value])
        plt.close()

        contour_points = []
        for collection in contour.collections:
            paths = collection.get_paths()
            for path in paths:
                contour_points.extend(path.vertices)

        return contour_points

    def plot_contour_points(contour_points):
        x, y = zip(*contour_points)
        plt.scatter(x, y, s=1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Contour Points')
        plt.show()

# Parametry normal inverse gamma
    x_min, x_max = 0.5, 3.0
    sigma_min, sigma_max = 1e-10, 1.0

contur=find_contour_points(f=normal_inverse_gamma, mu=mu, lambda_=lambda_, alpha=alpha, beta=beta, value=1.6, x_range=[x_min, x_max], y_range=[sigma_min, sigma_max], num_points=1000)
print(contur)
plot_contour_points(contur)
"""
import numpy as np
import scipy.stats as stats
from scipy.integrate import nquad

def normal_inverse_gamma(x, y, mu, lambda_, alpha, beta):
    inv_gamma = stats.invgamma(alpha, scale=beta)
    normal = stats.norm(mu, np.sqrt(y / lambda_))
    return normal.pdf(x) * inv_gamma.pdf(y)

def parametric_contour(t, contour_points):
    t = t % len(contour_points)
    t_prev = int(t)
    t_next = (t_prev + 1) % len(contour_points)
    t_frac = t - t_prev
    return contour_points[t_prev] * (1 - t_frac) + contour_points[t_next] * t_frac

def green_volume_integral(f, contour_points, mu, lambda_, alpha, beta, n_points=1000):
    def integrand(s, t):
        x, y = parametric_contour(t, contour_points)
        dx, dy = parametric_contour(t + s, contour_points) - parametric_contour(t, contour_points)
        return f(x, y, mu, lambda_, alpha, beta) * dx * dy

    s_bounds = (0, 1)
    t_bounds = (0, len(contour_points))

    return nquad(integrand, [s_bounds, t_bounds])[0]

contour_points = np.array([(x, y) for x, y
"""