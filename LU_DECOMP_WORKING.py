import sympy as sp
import numpy as np
import pprint 
def lu_decomposition(A):
    n = A.shape[0]
    L = sp.eye(n)
    U = sp.zeros(n, n)

    # Compute U
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

    # Compute L
    #for i in range(n):
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

# Define random normally distributed values of k and f and h 
np.random.seed(0)
k = np.random.normal(3, 1.5, size = 4)
k0, k1, k2, k3 = k
print("k0:", k0)

f = np.random.normal(3, 1.5, size = (4,1))
f0,f1,f2,f3 = f
print("f0:", f0)

h = np.random.normal(3,1.5, size = 1)[0]
print(h)


# Define the matrix A(alpha)
A = np.array([
    [(k1 + 2*k0)/h, -k1/h, 0, 0],
    [-k1/h, (k1 + k2)/h, -k2/h, 0],
    [0, -k2/h, (k2 + k3)/h, -k3/h],
    [0, 0, -k3/h, k3/h]
])
A = np.abs(A)
print(A)



# Perform LU decomposition
L, U = lu_decomposition(A)


pprint.pprint(L)
pprint.pprint(U)



# Solve Ly = f using forward substitution
def forward_substitution(L, f):
    y = np.zeros(f.shape)
    for i in range(f.shape[0]):
        y[i] = f[i] - sum(L[i, j] * y[j] for j in range(i))
    return y
print(f.shape)
y = forward_substitution(L, f)
print(y)

# Solve Uu = y using backward substitution
def backward_substitution(U, y):
    u = np.zeros(y.shape)
    for i in reversed(range(y.shape[0])):
        u[i] = (y[i] - sum(U[i, j] * u[j] for j in range(i+1, y.shape[0]))) / U[i, i]
    return u

u = backward_substitution(U, y)

print(u)
