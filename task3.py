import numpy as np

A = np.array([
    [1, 10, 1],
    [2, 0, 1],
    [3, 3, 2]
])
B = np.array([
    [0.4, 2.4, -1.4],
    [0.14, 0.14, -0.14],
    [-0.85, -3.8, 2.8]
])

I = np.eye(A.shape[0])
tolerance = 1e-6
max_iter = 100

for iter_num in range(max_iter):
    B_next = B @ (2 * I - A @ B)
    if np.linalg.norm(B_next - B, ord=np.inf) < tolerance:
        print(f"Converged in {iter_num + 1} iterations.")
        break
    B = B_next
else:
    print("Maximum iterations reached without convergence.")

print("Refined Inverse:")
print(B)
