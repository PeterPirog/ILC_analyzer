import matplotlib.pyplot as plt
import numpy as np
from ILC_analyzer.nigtools import normal_inverse_gamma
from matplotlib.path import Path

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

def generate_points(x_min, x_max, sigma_min, sigma_max, N):
    x_values = np.linspace(x_min, x_max, N)
    sigma_values = np.linspace(sigma_min, sigma_max, N)

    X, Sigma = np.meshgrid(x_values, sigma_values)

    return X.flatten(), Sigma.flatten()

def is_point_inside_contour(X, Y, contour_points):
    path = Path(contour_points)
    points = np.column_stack((X, Y))
    inside = path.contains_points(points).astype(int)
    return inside

if __name__ == "__main__":
    #xi, sigmai = 1.23, 0.36
    xi, sigmai = 1.5, 0.5

    mu = 1.5720
    lambda_ = 31.3454
    alpha = 5.2138
    beta = 1.3484



    # range to plot integration area
    x_range = (mu - 10, mu + 10)
    y_range = (0.01, 10)
    # points to integrate in area
    N_integral_axis=1000

    # area to find maximum value of function and define plot area
    N_max_search=1000 # number of points in single axis to find maximum value of NIG
    search_x_min=mu - 10
    search_x_max = mu + 10
    search_sigma_min=0.01
    search_sigma_max=5

    def calculate_integral2_in_contur(xi, sigmai, mu, lambda_, alpha, beta,search_x_min, search_x_max, N_max_search):
        # calculate level NIG(xi,sigmai, params)
        lvl = normal_inverse_gamma(xi, sigmai, mu, lambda_, alpha, beta)

        #Define ranges to find contur C(x,y)
        x_search_range = np.linspace(search_x_min, search_x_max, N_max_search)
        sigma_search_range = np.linspace(search_sigma_min, search_sigma_max, N_max_search)
        X, Sigma = np.meshgrid(x_search_range, sigma_search_range)
        Z = normal_inverse_gamma(X, Sigma, mu, lambda_, alpha, beta) - lvl

        return 1


    lvl = normal_inverse_gamma(xi, sigmai, mu, lambda_, alpha, beta)

    print(f'lvl={lvl}')
    # range to search maximum function
    x_search_range = np.linspace(search_x_min, search_x_max, N_max_search)
    sigma_search_range = np.linspace(search_sigma_min, search_sigma_max, N_max_search)

    X, Sigma = np.meshgrid(x_search_range, sigma_search_range)
    Z = normal_inverse_gamma(X, Sigma, mu, lambda_, alpha, beta) - lvl

    # find contour and integration box
    contour_points, (x_min, x_max), (sigma_min, sigma_max) = find_contour_points(normal_inverse_gamma, mu, lambda_, alpha, beta, lvl, x_range=(mu - 10, mu + 10), y_range=y_range)
    print((x_min, x_max), (sigma_min, sigma_max))

    plot_contour(X, Sigma, Z, x_min, x_max, sigma_min, sigma_max)

    # Prostokąt do całkowania jest w obszarze x_min, x_max, sigma_min, sigma_max
    box_area=(x_max-x_min)*(sigma_max-sigma_min)
    print(f'box_area={box_area}')


    X, Y = generate_points(x_min, x_max, sigma_min, sigma_max, N_integral_axis)

    inside = is_point_inside_contour(X, Y, contour_points)
    print(inside)

    dxdy = box_area / (N_integral_axis ** 2)
    integral = dxdy * np.sum(inside * normal_inverse_gamma(X, Y, mu, lambda_, alpha, beta))
    print(f'integral={integral}, dxdy={dxdy}, confidence level:{1-integral}')


"""
przekształć cały skrypt tak aby stworzyć funkcje której argumentami wymaganymi są:     xi, sigmai = 1.23, 0.36

    mu = 1.5720
    lambda_ = 31.3454
    alpha = 5.2138
    beta = 1.3484, celem jest wyliczyć wartośc integral i narysować wykres, inne zmienne funkcji ustaw jako opcjonalne z wartościami domyślnymi jak w skrypcie
    
    
    Dane:
    mu,lambda_,alpha, beta oraz punkt xi,sigmai
    
    STEP 1: find value lvl=normal_inverse_gamma
    STEP 2 find ranges  (BOX) where completed contour is inside
    
    
    
    napisz funkcję  python wyliczającą całkę podwójna funkcji  normal inverse-gamma(x, sigma, mu, lambda_, alpha, beta)  wyliczoną w obszarzegraniczoną krzywą C(x,sigma) taką, że  normal inverse-gamma(x, sigma, mu, lambda_, alpha, beta)=a dla każdej pary x,sigma należące do C, parametry sigma, mu, lambda_, alpha, beta, a są znane

"""