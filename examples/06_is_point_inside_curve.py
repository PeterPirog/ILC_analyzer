import numpy as np
import matplotlib.pyplot as plt

def C(x, y):
    return (x-3)**2 + (y-3)**2 - 9

def inside_closed_curve(xi, yi, C, epsilon=1e-5, n_points=10000):
    if abs(C(xi, yi)) <= epsilon:
        return "Punkt na krzywej"

    intersections = 0
    for t in np.linspace(0, 1, n_points):
        x = xi + t * (1 - xi)
        y = yi

        if abs(C(x, y)) <= epsilon:
            intersections += 1

    return intersections % 2 == 1

xi, yi = 2, 2
is_inside = inside_closed_curve(xi, yi, C)

x = np.linspace(0, 6, 400)
y = np.linspace(0, 6, 400)
X, Y = np.meshgrid(x, y)
Z = C(X, Y)

plt.figure()
plt.contour(X, Y, Z, levels=[0], colors='blue')

plt.scatter(xi, yi, color='red', label=f'({xi}, {yi})')
plt.legend()

plt.xlabel('x')
plt.ylabel('y')

plt.title(f"Punkt ({xi}, {yi}) {'znajduje się' if is_inside else 'nie znajduje się'} wewnątrz krzywej.")

plt.show()
