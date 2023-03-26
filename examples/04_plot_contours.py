import pandas as pd
import numpy as np
from ILC_analyzer.nigtools import NigDist
import matplotlib.pyplot as plt
import scipy.stats as stats
from shapely.geometry import Point, Polygon

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

def normal_inverse_gamma(x, y, mu, lambda_, alpha, beta):
 inv_gamma = stats.invgamma(alpha, scale=beta)
 normal = stats.norm(mu, np.sqrt(y / lambda_))
 return normal.pdf(x) * inv_gamma.pdf(y)


def plot_contour_points(contour_points):
       x, y = zip(*contour_points)
       plt.scatter(x, y, s=1)
       plt.xlabel('x')
       plt.ylabel('y')
       plt.title('Contour Points')
       plt.show()

if __name__ == "__main__":
    path_data = '../data/ILC_data.xlsx'

    # Load data from excel file
    df = pd.read_excel(path_data)
    x_i = df['Avg'].values
    sigma_i = df['std'].values

    # Create an instance of NigDist to calculate parameters of normal-inverse-gamma distribution
    ilc = NigDist(x=x_i, sigma=sigma_i,k=1)
    x_min, x_max = 0.5, 3.0
    sigma_min, sigma_max = 1e-10, 2.0

    # Print estimated parameters of NIG distribution
    print(f"Estimated parameters of NIG distribution: mu={ilc.mu:.4f}, lambda={ilc.lambda_:.4f}, alpha={ilc.alpha:.4f}, beta={ilc.alpha:.4f}")
    contur=find_contour_points(f=normal_inverse_gamma, mu=ilc.mu, lambda_=ilc.lambda_, alpha=ilc.alpha, beta=ilc.alpha, value=1.6, x_range=[x_min, x_max], y_range=[sigma_min, sigma_max], num_points=1000)
    print(contur)
    plot_contour_points(contur)

    poly = Polygon(contur)

    point = Point(1.55, 0.8)
    if poly.contains(point):
           print("Punkt (2,2) znajduje się wewnętrz obszaru.")
    else:
           print("Punkt (2,2) znajduje się na zewnątrz obszaru.")

"""

Tak, funkcja gęstości prawdopodobieństwa musi spełniać następujące warunki:

Musi być nieujemna dla każdej wartości zmiennych losowych x i y: f(x,y) >= 0.
Całka z funkcji gęstości prawdopodobieństwa musi wynosić 1 w całym obszarze zmiennej losowej: ∫∫ f(x,y) dxdy = 1.
Prawdopodobieństwo wystąpienia zmiennej losowej w dowolnym obszarze A jest równe całce podwójnej funkcji gęstości prawdopodobieństwa w tym obszarze: P((x,y) ∈ A) = ∫∫A f(x,y) dxdy.


from shapely.geometry import Point, Polygon

# Definiujemy krzywą zamkniętą
points = [(1, 1), (2, 3), (3, 2), (2, 1)]
poly = Polygon(points)

# Sprawdzamy, czy punkt (2,2) znajduje się wewnętrz obszaru
point = Point(2, 2)
if poly.contains(point):
    print("Punkt (2,2) znajduje się wewnętrz obszaru.")
else:
    print("Punkt (2,2) znajduje się na zewnątrz obszaru.")

"""

"""
w jaki sposób wyliczyć całke potrójną  (objetość) funkcji z=f(x,y), gdzie x,y znajdują się w obszarze ograniczoną krzywą zamknietą, kształt krzywej zdefiniowano w postaci par wspłrzędnych (xi,yi)
"""