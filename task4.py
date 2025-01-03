import numpy as np

A = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
])

v = np.array([1, 0, 0], dtype=float)


tolerance = 1e-6
max_iter = 100


lambda_old = 0
for i in range(max_iter):

    w = np.dot(A, v)

    v = w / np.linalg.norm(w)

    lambda_new = np.dot(v.T, np.dot(A, v))

    if abs(lambda_new - lambda_old) < tolerance:
        break
    lambda_old = lambda_new


print(f"eigenvalue: {lambda_new}")
print(f"eigenvector: {v}")
