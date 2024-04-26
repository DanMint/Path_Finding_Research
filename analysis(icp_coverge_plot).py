import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)

    T = np.identity(len(A[0])+1)
    T[:len(A[0]), :len(A[0])] = R
    T[:len(A[0]), len(A[0])] = t

    return T, R, t

def icp(A, B, init_pose=None, max_iterations=100, tolerance=0.001):
    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    if init_pose is not None:
        src = np.dot(init_pose, src)

    iteration_errors = []

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        valid = distances < 1.0
        if np.sum(valid) == 0:
            break

        T, _, _ = best_fit_transform(src[:m, valid].T, dst[:m, indices[valid]].T)
        src = np.dot(T, src)

        mean_error = np.mean(distances[valid])
        iteration_errors.append(mean_error)

        if np.abs(mean_error) < tolerance:
            break

    T, _, _ = best_fit_transform(A, src[:m,:].T)
    return T, iteration_errors, i

def plot_icp_convergence(errors):
    plt.figure(figsize=(10, 5))
    plt.plot(errors, marker='o', linestyle='-', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Error')
    plt.title('ICP Convergence Plot')
    plt.grid(True)
    plt.show()

def load_point_sets(filename):
    A = np.random.rand(100, 2)  
    B = np.random.rand(100, 2)  
    return A, B

A, B = load_point_sets('intel.clf')
T, errors, iterations = icp(A, B)
plot_icp_convergence(errors)
