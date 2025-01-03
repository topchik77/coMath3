import numpy as np

def jacobi_eigen(A, tol=1e-6, max_iter=100):

    if not np.allclose(A, A.T, atol=tol):
        raise ValueError("Matrix must be symmetric.")

    n = A.shape[0]
    V = np.eye(n)
    
    for _ in range(max_iter):

        i, j = np.unravel_index(np.argmax(np.abs(A - np.diag(np.diag(A)))), A.shape)
        if abs(A[i, j]) < tol:
            break
        

        theta = 0.5 * np.arctan2(2 * A[i, j], A[j, j] - A[i, i])
        

        R = np.eye(n)
        R[i, i] = R[j, j] = np.cos(theta)
        R[i, j] = -np.sin(theta)
        R[j, i] = np.sin(theta)

        A = R.T @ A @ R
        V = V @ R
    
    eigenvalues = np.diag(A)
    return eigenvalues, V

A = np.array([
    [1, np.sqrt(2), 2],
    [np.sqrt(2), 3, np.sqrt(2)],
    [2, np.sqrt(2), 1]
])

eigenvalues, eigenvectors = jacobi_eigen(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
