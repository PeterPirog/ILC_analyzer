import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ILC_analyzer.nigtools import NigDist, normal_inverse_gamma


def plot_2d(ilc, level):
    """
    Plots a 2D contour plot of the function defined by Z over the (x, sigma) domain.
    """
    x, sigma = np.meshgrid(ilc.x_range, ilc.sigma_range)
    Z = normal_inverse_gamma(x, sigma, ilc.mu, ilc.lambda_, ilc.alpha, ilc.beta)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.contourf(x, sigma, Z, cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('sigma')
    plt.title('2D Plot with Color')
    plt.colorbar()
    plt.scatter(ilc.xi, ilc.sigmai, marker='+', color='black')

    # Add probability contour to 2D plot
    cs = ax.contour(x, sigma, Z, levels=level, colors='red')

    # Add labels with index for every ilc.xi point
    for i, (x, sigma) in enumerate(zip(ilc.xi, ilc.sigmai)):
        ax.annotate(f'{i}', (x, sigma), textcoords='offset points', xytext=(0, 5), ha='center')

    plt.show()

    return cs


def calculate_volume(cs, ilc, num_points=10000):
    """
    Calculates the volume in space between the contour lines defined by cs using Monte Carlo method.
    """
    # Get bounding box of contour lines
    bounds = cs.allsegs[0][0].copy()
    for path in cs.allsegs[0][1:]:
        bounds = np.concatenate((bounds, path))

    x_min, y_min = bounds.min(axis=0)
    x_max, y_max = bounds.max(axis=0)

    # Generate random points inside the bounding box
    x_points = np.random.uniform(x_min, x_max, num_points)
    y_points = np.random.uniform(y_min, y_max, num_points)

    # Calculate the fraction of points inside the contour lines
    count_inside = 0
    for i in range(num_points):
        x, y = x_points[i], y_points[i]
        if any(cs.contains_point([x, y]) for cs in cs.collections):
            count_inside += 1

    fraction_inside = count_inside / num_points

    # Calculate the volume using the fraction of points inside the contour lines
    volume = (x_max - x_min) * (y_max - y_min) * fraction_inside

    return volume


# Load data from Excel file
path_data = '../data/ILC_data.xlsx'
df = pd.read_excel(path_data)

# Extract data from dataframe
x_i = df['Avg'].values
sigma_i = df['std'].values

# Create an instance of NigDist to calculate parameters of normal-inverse-gamma distribution
ilc = NigDist(x=x_i, sigma=sigma_i, k=1)

# Print estimated parameters of NIG distribution
print(
    f"Estimated parameters of NIG distribution: mu={ilc.mu:.4f}, lambda={ilc.lambda_:.4f}, alpha={ilc.alpha:.4f}, beta={ilc.beta:.4f}")

# Set contour level
level = [1.6]

# Plot 2D contour plot and get the contour object
cs = plot_2d(ilc, level)
print(cs.allsegs)
volume = calculate_volume(cs, ilc)
print(volume)
"""
mając daną funkcję dodatnią f(x,y) >0, i kontur krzywej zamknietej zdefiniowanej parami punktów (x,y)
napisz funkcję w python wyznaczającą objetość przestrzeni pomiędzy funkcją f(x,y) dla wszystkich x,y znajdujących się wewnątz konturu
a płaszczyzną f(x,y)=0
"""

