import numpy as np
from scipy.linalg import lu, solve

A = np.array([
    [50, 107, 36],
    [35, 54, 20],
    [31, 66, 21]
])


P, L, U = lu(A)


I = np.eye(3)
inverse = np.zeros_like(A, dtype=float)

for i in range(3):

    Y = solve(L, I[:, i])

    X = solve(U, Y)
    inverse[:, i] = X

print("LU Inverse:")
print(inverse)
