import numpy as np

def thomasalg(a, b, c, r):
    n = len(b)
    c_prime = np.zeros(n - 1)
    r_prime = np.zeros(n)

    # Forward elimination of first row
    c_prime[0] = c[0] / b[0]
    r_prime[0] = r[0] / b[0]

    # Forward elimination
    for i in range(1, n):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        r_prime[i] = (r[i] - r_prime[i - 1] * a[i - 1]) / denom
        if i < n - 1:
            c_prime[i] = c[i] / denom

    # Backward substitution
    x = np.zeros(n)
    x[-1] = r_prime[-1]

    for i in range(n - 2, -1, -1):
        x[i] = r_prime[i] - c_prime[i] * x[i + 1]

    return x
