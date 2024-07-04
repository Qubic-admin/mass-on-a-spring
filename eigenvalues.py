import numpy as np
from numpy import linalg as LA

# Parameter values
# Masses:
m1 = 1.5
m2 = 1.5
# Spring constants
k1 = 2.0
k2 = 2.0
# # Friction coefficients
# b1 = 0.1
# b2 = 0.1
# res freq
om1 = 2.0
om2 = 2.0
# # gamma
# v1 = 0.5
# v2 = 0.5
# kc
kc = 3.0


# plot over range of kc

f1 = np.array([[-m1 * om1**2 + k1 + kc, -kc], [-kc, -m2 * om2**2 + k2 + kc]])
f2 = np.array(
    [
        [0, 1, 0, 0],
        [-m1 * om1**2 + k1 + kc, 0, -kc, 0],
        [0, 0, 0, 1],
        [-kc, 0, -m2 * om2**2 + k2 + kc, 0],
    ]
)
eigenvalues, eigenvectors = LA.eig(f1)
print(eigenvalues)
