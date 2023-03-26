import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from shapely.geometry import Point, Polygon


def f(x, y, a):
    return x * y * a

def plot_function(contour_points, X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax.plot(contour_points[:, 0], contour_points[:, 1], zs=0, c='r', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def inside_polygon(x, y):
    return polygon.contains(Point(x, y))

def integration_function(y, x, a):
    return f(x, y, a) if inside_polygon(x, y) else 0
if __name__ == '__main__':
    contour_points = np.array([(1.5, 3), (3, 2), (3, 4), (1, 5), (1.5, 3)])
    polygon = Polygon(contour_points[:-1])

    x_min, y_min, x_max, y_max = polygon.bounds
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    a = 2
    Z = f(X, Y, a)
    plot_function(contour_points, X, Y, Z)

    result, error = dblquad(lambda y, x: integration_function(y, x, a), x_min, x_max, lambda x: y_min, lambda x: y_max, epsabs=1e-6, epsrel=1e-6)

    print("Wynik całkowania:", result)
    print("Błąd numeryczny:", error)
