import matplotlib.pyplot as plt
import numpy as np
from ILC_analyzer.nigtools import normal_inverse_gamma

def find_contour_points(f, mu, lambda_, alpha, beta, value, x_range, y_range, num_points=1000):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x_values, y_values)
    Z = f(X, Y, mu, lambda_, alpha, beta) - value

    contour = plt.contour(X, Y, Z, levels=[0])
    plt.close()

    contour_points = []
    for collection in contour.collections:
        paths = collection.get_paths()
        for path in paths:
            contour_points.extend(path.vertices)

    x_min = min(contour_points, key=lambda x: x[0])[0]
    x_max = max(contour_points, key=lambda x: x[0])[0]
    sigma_min = min(contour_points, key=lambda x: x[1])[1]
    sigma_max = max(contour_points, key=lambda x: x[1])[1]

    return contour_points, (x_min, x_max), (sigma_min, sigma_max)

def plot_contour(X, Sigma, Z, x_min, x_max, sigma_min, sigma_max):
    plt.contour(X, Sigma, Z, levels=[0], colors='blue')
    plt.fill([x_min, x_min, x_max, x_max], [sigma_min, sigma_max, sigma_max, sigma_min], 'r', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('sigma')
    plt.title('Kontur C(x, sigma) dla pdf(x, sigma | mu, lambda_, alpha, beta) = lvl')
    plt.show()

if __name__ == "__main__":
    #xi, sigmai = 1.23, 0.36
    xi, sigmai = 2, 2

    mu = 1.5720
    lambda_ = 31.3454
    alpha = 5.2138
    beta = 1.3484
    # range to plot inegration area
    x_range = (mu - 10, mu + 10)
    y_range = (0.01, 10)


    lvl = normal_inverse_gamma(xi, sigmai, mu, lambda_, alpha, beta)

    print(f'lvl={lvl}')
    # range to search maximum function
    x_search_range = np.linspace(mu - 10, mu + 10, 1000)
    sigma_search_range = np.linspace(0.01, 10, 1000)

    X, Sigma = np.meshgrid(x_search_range, sigma_search_range)
    Z = normal_inverse_gamma(X, Sigma, mu, lambda_, alpha, beta) - lvl

    # find contur and integration box
    contour_points, (x_min, x_max), (sigma_min, sigma_max) = find_contour_points(normal_inverse_gamma, mu, lambda_, alpha, beta, lvl, x_range=(mu - 10, mu + 10), y_range=y_range)
    print((x_min, x_max), (sigma_min, sigma_max))

    plot_contour(X, Sigma, Z,x_min, x_max, sigma_min, sigma_max)

    # Prostokąt do całkowania jest w obszarze x_min, x_max, sigma_min, sigma_max
    box_area=(x_max-x_min)*(sigma_max-sigma_min)
    print(f'box_area={box_area}')
    N=100


    def generate_points(x_min, x_max, sigma_min, sigma_max, N):
        x_values = np.linspace(x_min, x_max, N)
        sigma_values = np.linspace(sigma_min, sigma_max, N)

        X, Y = np.meshgrid(x_values, sigma_values)

        X = X.flatten()
        Y = Y.flatten()

        return X, Y


    X, Y = generate_points(x_min, x_max, sigma_min, sigma_max, N)

    print(X)
    print(Y)

    from matplotlib.path import Path


    def is_point_inside_contour(X, Y, contour_points):
        path = Path(contour_points)
        inside = np.zeros_like(X)
        for i in range(len(X)):
            inside[i] = int(path.contains_point([X[i], Y[i]]))
        return inside


    inside=is_point_inside_contour(X, Y, contour_points)
    print(inside)

    #def integral_in_contur(func,X,Y,contour_points):

    dxdy=box_area/(N**2)
    integral=dxdy*np.sum(inside*normal_inverse_gamma(X, Y, mu, lambda_, alpha, beta))
    print(f'integral={integral}, dxdy={dxdy}')