from ILC_analyzer.nigtools import normal_inverse_gamma
import numpy as np
import pandas as pd
import numpy as np
from ILC_analyzer.nigtools import NigDist
import matplotlib.pyplot as plt
import scipy.stats as stats
from shapely.geometry import Point, Polygon


def plot_contour_points(contour_points):
    x, y = zip(*contour_points)
    plt.scatter(x, y, s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour Points')
    plt.show()
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

#Estimated parameters of NIG distribution:
mu = 1.5720
lambda_ = 31.3454
alpha = 5.2138
beta = 1.3484

# Parametry normal inverse gamma
num_points = 1000 # value  for finding NIG range
x_min, x_max = 0.5, 3.0
sigma_min, sigma_max = 1e-10, 1.0

x_values = np.linspace(x_min, x_max, num_points)
sigma_values = np.linspace(sigma_min, sigma_max, num_points)
X, SIGMA = np.meshgrid(x_values, sigma_values)
Z = normal_inverse_gamma(X, SIGMA, mu, lambda_, alpha, beta)

max_idx = np.unravel_index(np.argmax(Z), Z.shape)
x_mod = X[max_idx]
sigma_mod = SIGMA[max_idx]
z_min=0.0
z_max=np.max(Z)


print("Min Z: ", np.min(Z))
print("Max Z: ", np.max(Z))
print("Max Z index: ", max_idx)
print("x_mod: ", x_mod)
print("sigma_mod: ", sigma_mod)

# generowanie punktów N wymiaroowych
N_3D=100 # ^3
value=1.0


x = np.linspace(x_min, x_max, N_3D)
y = np.linspace(sigma_min, sigma_max, N_3D)
z = np.linspace(z_min, z_max, N_3D)

# utworzenie siatki z punktami w przestrzeni 3D do całkowania numerycznego
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
print(points, np.shape(points))


# całkowita objetośc obszaru
volume_total=x_max*np.abs( x_max-x_min)*np.abs( sigma_max-sigma_min)
print(volume_total)

# szukanie kontury dla poziomu
sigma_max=1.5
contur = find_contour_points(f=normal_inverse_gamma, mu=mu, lambda_=lambda_, alpha=alpha, beta=alpha,
                             value=value, x_range=[x_min, x_max], y_range=[sigma_min, sigma_max], num_points=1000)
print(contur)
#plot_contour_points(contur)

poly = Polygon(contur)

point_list=[]
for point in points:
    x=point[0]
    y=point[1]
    z=point[2]
    if poly.contains(Point(x, y)) and (z <=normal_inverse_gamma(x, y, mu, lambda_, alpha, beta)):
        point_list.append(point)

print(point_list, len(point_list),len(points))
volume=len(point_list)*volume_total/len(points)
print(volume)



####################################
"""
point = Point(1.55, 0.8)
if poly.contains(point):
    print("Punkt (2,2) znajduje się wewnętrz obszaru.")
else:
    print("Punkt (2,2) znajduje się na zewnątrz obszaru.")

x=1.57
sigma=0.25

z=normal_inverse_gamma(x, sigma, mu, lambda_, alpha, beta)
print(z)
"""
