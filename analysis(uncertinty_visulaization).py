import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

def plot_covariance_ellipse(x, y, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=(x, y), width=width, height=height, angle=angle, facecolor=facecolor, edgecolor='red', linewidth=2, alpha=0.5, **kwargs)
    ax.add_patch(ellipse)

def read_odom_data(file_path):
    poses = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('ODOM'):
                parts = line.split()
                x = float(parts[1])
                y = float(parts[2])
                poses.append((x, y))
    return poses

poses = read_odom_data('intel.clf')  
fig, ax = plt.subplots()
for x, y in poses:
    cov = np.array([[0.05, 0.0], [0.0, 0.05]])
    plot_covariance_ellipse(x, y, cov, ax)

ax.set_aspect('equal')
ax.set_xlim(min([p[0] for p in poses]) - 1, max([p[0] for p in poses]) + 1)
ax.set_ylim(min([p[1] for p in poses]) - 1, max([p[1] for p in poses]) + 1)
plt.show()
